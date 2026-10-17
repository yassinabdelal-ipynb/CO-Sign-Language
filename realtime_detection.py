"""Checkpoint-compatible EyeViT and an explicit, configurable eye-event language decoder.
Reconstruction: this file does not recover the original study's language mapping.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from isef_preprocess import training_array, collect_eye, EyeLocator, roi_crop

class Patches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, images):
        # NCHW -> [batch, row, col, patch_row, patch_col, channel].
        # This matches tf.image.extract_patches pixel order, including channels.
        p = self.patch_size
        tiles = images.unfold(2, p, p).unfold(3, p, p)
        return tiles.permute(0, 2, 3, 4, 5, 1).contiguous().reshape(
            images.shape[0], -1, p * p * images.shape[1]
        )

class PatchEncoder(nn.Module):
    def __init__(self, num_patches, projection_dim, patch_dim):
        super().__init__()
        self.projection = nn.Linear(patch_dim, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)

    def forward(self, patches):
        positions = torch.arange(patches.shape[1], device=patches.device)
        return self.projection(patches) + self.position_embedding(positions)

class KerasStyleAttention(nn.Module):
    def __init__(self, width, heads, key_dim):
        super().__init__()
        self.heads, self.key_dim = heads, key_dim
        # Original Keras key_dim=64 means 64 PER head: total inner width=512.
        self.query = nn.Linear(width, heads * key_dim)
        self.key = nn.Linear(width, heads * key_dim)
        self.value = nn.Linear(width, heads * key_dim)
        self.output = nn.Linear(heads * key_dim, width)

    def forward(self, x):
        b, n, _ = x.shape
        def project(layer):
            return layer(x).reshape(b, n, self.heads, self.key_dim).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            project(self.query), project(self.key), project(self.value),
            dropout_p=0.0, is_causal=False
        )
        return self.output(attended.transpose(1, 2).contiguous().reshape(b, n, -1))

    def reset_keras_parameters(self):
        # Match Glorot's fan calculation for Keras 3-D attention kernels.
        width, heads, dim = self.query.in_features, self.heads, self.key_dim
        qkv_bound = math.sqrt(6.0 / (width * heads + width * dim))
        out_bound = math.sqrt(6.0 / (heads * dim + heads * width))
        for layer in (self.query, self.key, self.value):
            nn.init.uniform_(layer.weight, -qkv_bound, qkv_bound)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.output.weight, -out_bound, out_bound)
        nn.init.zeros_(self.output.bias)

def mlp(in_features, hidden_units, dropout):
    layers = []
    for units in hidden_units:
        layers.extend([nn.Linear(in_features, units), nn.GELU(approximate='none'), nn.Dropout(dropout)])
        in_features = units
    return nn.Sequential(*layers)

class TransformerBlock(nn.Module):
    def __init__(self, width, heads, hidden_units, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attention = KerasStyleAttention(width, heads, key_dim=width)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = mlp(width, hidden_units, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))

class EyeViT(nn.Module):
    def __init__(self, image_size=224, patch_size=7, projection_dim=64,
                 num_heads=8, transformer_layers=12, mlp_head_units=(128, 64),
                 num_classes=6, dropout_rate=0.1):
        super().__init__()
        if mlp_head_units[-1] != projection_dim:
            raise ValueError("Residual MLP must end at projection_dim.")
        self.patches = Patches(patch_size)
        self.encoder = PatchEncoder((image_size // patch_size)**2, projection_dim, patch_size**2)
        self.blocks = nn.Sequential(*[
            TransformerBlock(projection_dim, num_heads, mlp_head_units, dropout_rate)
            for _ in range(transformer_layers)
        ])
        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.head = mlp(projection_dim, mlp_head_units, dropout_rate)
        self.classifier = nn.Linear(mlp_head_units[-1], num_classes)
        self.apply(self._initialize)
        for module in self.modules():
            if isinstance(module, KerasStyleAttention):
                module.reset_keras_parameters()

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.uniform_(module.weight, -0.05, 0.05)

    def forward(self, images):
        x = self.encoder(self.patches(images))
        x = self.norm(self.blocks(x)).mean(dim=1)
        # CrossEntropyLoss includes log-softmax; apply softmax only for predictions.
        return self.classifier(self.head(x))


def read_mapping(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported mapping schema')
    codes = data.get('commands', {})
    if not isinstance(codes, dict) or not codes:
        raise ValueError('Mapping must contain commands')
    for code, output in codes.items():
        if code != ' '.join(code.split()) or len(code.split()) not in (3, 4) or any(d not in {'up','down','left','right'} for d in code.split()):
            raise ValueError('Every command must contain exactly three or four directions')
        if not isinstance(output, str) or not output.strip():
            raise ValueError('Command output must be nonempty text')
        if any(other.startswith(code+' ') for other in codes):
            raise ValueError('A complete command cannot be a prefix of another command')
    required = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') | {"I'm hungry", "I'm thirsty", "I'm fine"}
    if set(codes.values()) != required or len(codes) != len(required):
        raise ValueError('Mapping must cover A-Z and the three phrases exactly once')
    return codes


class Decoder:
    """Dwell -> event -> prefix-free command. Center dwell re-arms each event.

    Center/close are not command directions. Missing/low-confidence frames
    interrupt a dwell and never re-arm it. Timeout clears incomplete commands.
    These rules are engineering debounce rules, not an intention classifier.
    """
    def __init__(self, mapping, dwell=.35, neutral=.15, timeout=5., confidence=.8):
        if not all(math.isfinite(x) and x > 0 for x in (dwell, neutral, timeout)) or timeout <= dwell:
            raise ValueError('Positive finite dwell/neutral and timeout > dwell required')
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Confidence must be between zero and one')
        self.mapping = mapping
        self.dwell, self.neutral, self.timeout, self.confidence = dwell, neutral, timeout, confidence
        self.sequence = []
        self.label = None
        self.since = None
        self.previous_time = None
        self.last_event = None
        self.armed = False  # Must establish center before accepting first event.

    def reset(self):
        self.sequence.clear()
        self.label = self.since = self.last_event = None
        self.armed = False

    def update(self, label, score, now):
        if not math.isfinite(now) or (self.previous_time is not None and now < self.previous_time):
            raise ValueError('Timestamps must be finite and monotonic')
        # Never carry a dwell across a camera stall or sparse event stream.
        if self.previous_time is not None and now-self.previous_time > .25:
            self.label = self.since = None
            self.armed = False
        self.previous_time = now
        if self.last_event is not None and now-self.last_event > self.timeout:
            self.reset()
        if not math.isfinite(score) or not self.confidence <= score <= 1:
            self.label = self.since = None
            return None
        if label not in {'center','up','down','left','right'}:
            self.label = self.since = None
            return None
        if label != self.label:
            self.label, self.since = label, now
        elapsed = now-self.since
        if label == 'center':
            if elapsed >= self.neutral:
                self.armed = True
            return None
        if not self.armed or elapsed < self.dwell:
            return None
        self.armed = False
        self.sequence.append(label)
        self.last_event = now
        code = ' '.join(self.sequence)
        if code in self.mapping:
            self.sequence.clear()
            return self.mapping[code]
        if not any(k.startswith(code+' ') for k in self.mapping):
            self.sequence.clear()
        return None


def load_model(path, device):
    if not Path(path).is_file():
        raise FileNotFoundError('Checkpoint missing. Supply a trusted notebook-produced .pt; no weights are bundled or downloaded.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not {'model_config','class_names','model_state_dict'} <= checkpoint.keys():
        raise ValueError('Checkpoint needs model_config, class_names and model_state_dict')
    config, names = checkpoint['model_config'], checkpoint['class_names']
    if not isinstance(names, list) or len(set(names)) != len(names) or set(names) != {'center','close','up','down','left','right'}:
        raise ValueError('Expected the six notebook classes, using checkpoint order')
    if config.get('num_classes') != len(names) or config.get('image_size') != 224 or config.get('patch_size') != 7:
        raise ValueError('Unsupported checkpoint model/input contract')
    preprocessing = checkpoint.get('preprocessing', {})
    if preprocessing.get('grayscale') is not True or preprocessing.get('resize') != [224,224] or not math.isclose(preprocessing.get('rescale',0),1/255):
        raise ValueError('Checkpoint preprocessing does not match grayscale/nearest224/rescale1/255')
    model = EyeViT(**config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Checkpoint contains non-finite parameters')
    return model.to(device).eval(), names


def predict(model, image, device):
    with torch.inference_mode():
        scores = model(torch.from_numpy(training_array(image)).to(device)).softmax(dim=-1)[0].cpu().numpy()
    if scores.shape != (6,) or not np.isfinite(scores).all():
        raise ValueError('Invalid inference output')
    return scores


def replay(path, decoder):
    emitted = []
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
                result = decoder.update(event['label'], float(event['confidence']), float(event['time']))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Invalid event at line {number}: {exc}') from exc
            if result:
                emitted.append(result)
    return emitted


def main():
    parser = argparse.ArgumentParser(description='CO-Sign: local eye classifier and experimental command decoder')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--image', type=Path, help='Already cropped eye; no camera or locator needed')
    parser.add_argument('--events', type=Path, help='Replay labeled events; synthetic examples are not model accuracy tests')
    parser.add_argument('--mapping', type=Path, default=Path(__file__).parent/'config/language.json')
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--predictor', type=Path)
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X','Y','WIDTH','HEIGHT'), help='Explicit manual crop; approximate geometry')
    parser.add_argument('--eye', choices=['image-left','image-right'], default='image-left')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--seconds', type=float, default=300, help='Bound camera session duration')
    parser.add_argument('--theme', choices=['dark','light'], default='dark')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be positive and finite')
    decoder = Decoder(read_mapping(args.mapping))
    if args.events:
        print(json.dumps({'source':'event replay; not measured model predictions','emitted':replay(args.events, decoder)}))
        return 0
    if not args.checkpoint:
        parser.error('--checkpoint is required for inference; use --events for a no-camera demonstration')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable. Use --device cpu or inspect the CUDA setup.')
    model, names = load_model(args.checkpoint, args.device)
    if args.image:
        with Image.open(args.image) as image:
            scores = predict(model, image, args.device)
        print(json.dumps({'scores':dict(zip(names, map(float,scores))), 'note':'Prediction only; no ground-truth accuracy inferred'}))
        return 0
    if not args.predictor and not args.roi:
        parser.error('Camera mode requires --predictor PATH or an explicitly approximate --roi X Y W H')
    import cv2
    locator = EyeLocator(args.predictor) if args.predictor else None
    cap = None
    try:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError('Camera unavailable. Close other camera applications or choose --camera 1.')
        cap.set(cv2.CAP_PROP_FPS, 30)
        start, text = time.monotonic(), ''
        title = 'CO-Sign | experimental decoder | Q quit'
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        while time.monotonic()-start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera stopped returning frames')
            points = locator.locate(frame, args.eye) if locator else None
            image = collect_eye(frame, points) if points is not None else (roi_crop(frame,args.roi) if args.roi else None)
            label, confidence = 'no face', 0.
            if image is not None:
                scores = predict(model, image, args.device)
                index = int(scores.argmax())
                label, confidence = names[index], float(scores[index])
            result = decoder.update(label, confidence, time.monotonic())
            if result:
                text += (' '+result+' ') if len(result)>1 else result
            bg, fg = ((28,28,28),(235,235,235)) if args.theme == 'dark' else ((250,250,250),(25,25,25))
            panel = np.empty((290,max(800,frame.shape[1]),3),dtype=np.uint8)
            panel[:] = bg
            lines = [f'{label} {confidence:.1%} | {args.device} | new demonstration mapping',
                     'Pending: '+' '.join(decoder.sequence), 'Text: '+text[-75:],
                     'Hold center, then each direction. Return to center between events.',
                     'Space: space | Backspace: delete | C: clear | Q/Esc: quit',
                     'No intention filter active. close is a class, not a timed blink detector.']
            for row,line in enumerate(lines):
                cv2.putText(panel,line,(12,32+42*row),cv2.FONT_HERSHEY_SIMPLEX,.55,fg,1,cv2.LINE_AA)
            canvas=np.full((frame.shape[0]+290,panel.shape[1],3),bg,dtype=np.uint8)
            canvas[:frame.shape[0],:frame.shape[1]]=frame
            canvas[frame.shape[0]:]=panel
            cv2.imshow(title,canvas)
            key=cv2.waitKey(1)&255
            if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                break
            if key == ord('c'): text=''; decoder.reset()
            elif key == 32: text+=' '
            elif key == 8: text=text[:-1]; decoder.reset()
    finally:
        if cap is not None: cap.release()
        if locator is not None: locator.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'CO-Sign: {exc}', file=sys.stderr)
        raise SystemExit(2)


