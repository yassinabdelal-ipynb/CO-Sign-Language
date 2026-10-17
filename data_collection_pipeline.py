"""Bounded local collection: camera -> face -> landmarks -> eye -> gate -> PNG.

No raw face frames are saved. All outputs remain private and Git-ignored.
YOLO requires the author's trained open/closed checkpoint, which is not supplied.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import uuid
import numpy as np
from PIL import Image
from isef_preprocess import EyeLocator, collect_eye, augment, keep_eye, PROFILE


class YoloEyeGate:
    def __init__(self, weights, confidence=.8):
        if not weights or not Path(weights).is_file():
            raise FileNotFoundError('Supply local --eye-weights for a trained open/closed YOLO. Generic COCO weights are not suitable. No download is attempted.')
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.confidence = confidence
        self.names = {int(k): str(v).lower() for k,v in self.model.names.items()}
        if set(self.names.values()) != {'open','closed'}:
            raise ValueError('Eye-state checkpoint must have exactly open and closed classes')
        if self.model.task not in {'classify','detect'}:
            raise ValueError('Eye-state model must be a classifier or detector')

    def __call__(self, image):
        # Pass an explicit local image; do not invoke implicit sample-image sources.
        result = self.model.predict(source=image.convert('RGB'), verbose=False, save=False)[0]
        if result.probs is not None:
            p = result.probs
            return self.names[int(p.top1)] if float(p.top1conf) >= self.confidence else 'unknown'
        if result.boxes is None or len(result.boxes) == 0:
            return 'unknown'
        states = [self.names[int(c)] for c,s in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()) if s >= self.confidence]
        # Conflicting detections cannot establish the state of this single eye.
        return states[0] if states and len(set(states)) == 1 else 'unknown'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictor', type=Path, required=True)
    parser.add_argument('--eye-weights', type=Path)
    parser.add_argument('--gate', choices=['yolo','manual'], default='yolo')
    parser.add_argument('--manual-state', choices=['open','closed','unknown'], default='unknown', help='A user declaration, not a measurement')
    parser.add_argument('--label', choices=['up','down','left','right','center','blink'], required=True)
    parser.add_argument('--eye', choices=['image-left','image-right'], default='image-left')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--fps', type=float, default=30)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--output', type=Path, default=Path('data/collected'))
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--augment', action='store_true', help='Also gate/store a Gaussian-noise derivative; never flip direction labels')
    parser.add_argument('--consent', action='store_true', help='Confirm permission to collect this participant’s eye images')
    args = parser.parse_args()
    if not args.consent:
        parser.error('Eye images can identify people. Obtain permission, then pass --consent.')
    if not math.isfinite(args.fps) or not 0 < args.fps <= 120 or args.max_frames <= 0:
        parser.error('Require 0 < fps <= 120 and positive max-frames')
    # Fail before opening a camera or creating a session if required assets are missing.
    gate = YoloEyeGate(args.eye_weights) if args.gate == 'yolo' else lambda image: args.manual_state
    locator = EyeLocator(args.predictor)
    import cv2
    rng = np.random.default_rng(args.seed)
    session = args.output / uuid.uuid4().hex
    session.mkdir(parents=True, exist_ok=False)
    counts = {'frames':0,'no_face':0,'retained':0,'rejected':0}
    hashes = {'predictor':hashlib.sha256(args.predictor.read_bytes()).hexdigest()}
    if args.eye_weights:
        hashes['eye_weights'] = hashlib.sha256(args.eye_weights.read_bytes()).hexdigest()
    metadata = {'schema_version':1,'session':session.name,'profile':PROFILE,'label':args.label,
        'eye':args.eye,'gate':args.gate,'manual_state':args.manual_state if args.gate=='manual' else None,
        'seed':args.seed,'requested_fps':args.fps,'max_frames':args.max_frames,'asset_sha256':hashes,
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','Pillow','opencv-contrib-python','mediapipe','dlib']},
        'status':'running','note':'Frame rate is requested, not guaranteed. Manual state is not inferred eye state.'}
    (session/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    cap = None
    started = time.monotonic()
    title = 'CO-Sign collection | '+args.gate+' gate | Q stop'
    try:
        with (session/'frames.jsonl').open('x',encoding='utf-8') as manifest:
            cap = cv2.VideoCapture(args.camera,cv2.CAP_DSHOW) if sys.platform=='win32' else cv2.VideoCapture(args.camera)
            if not cap.isOpened():
                raise RuntimeError('Camera unavailable; no images collected')
            cap.set(cv2.CAP_PROP_FPS,args.fps)
            metadata['camera_reported_fps'] = cap.get(cv2.CAP_PROP_FPS)
            while counts['frames'] < args.max_frames:
                tick = time.monotonic()
                ok,frame = cap.read()
                if not ok: raise RuntimeError('Camera stopped returning frames')
                index = counts['frames']; counts['frames'] += 1
                points = locator.locate(frame,args.eye)
                records = []
                if points is None:
                    counts['no_face'] += 1
                    records.append({'source_frame':index,'status':'no_face'})
                else:
                    raw = collect_eye(frame,points)
                    candidates = [('original',raw)]
                    if args.augment: candidates.append(('noise',augment(raw,rng)))
                    for kind,image in candidates:
                        state = gate(image)
                        retained = keep_eye(args.label,state)
                        record = {'source_frame':index,'elapsed_s':time.monotonic()-started,
                            'variant':kind,'state':state,'gate':args.gate,'retained':retained,'label':args.label}
                        counts['retained' if retained else 'rejected'] += 1
                        if retained:
                            filename = f'{index:06d}_{kind}.png'
                            image.save(session/filename)
                            record['file'] = filename
                        records.append(record)
                for record in records: manifest.write(json.dumps(record)+'\n')
                manifest.flush()
                preview=frame.copy()
                cv2.putText(preview,f'{args.label} | {args.gate} gate | kept {counts["retained"]} | Q stop',
                    (10,30),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
                cv2.imshow(title,preview)
                key=cv2.waitKey(1)&255
                if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                    metadata['status']='cancelled'; break
                # Bound capture requests; inference can reduce processed rate below target.
                time.sleep(max(0.,1/args.fps-(time.monotonic()-tick)))
            else: metadata['status']='completed'
    except KeyboardInterrupt:
        metadata['status']='cancelled'
    except Exception as exc:
        metadata['status']='failed'; metadata['error']=str(exc)
        raise
    finally:
        if cap is not None: cap.release()
        locator.close()
        cv2.destroyAllWindows()
        metadata.update(counts=counts,elapsed_s=time.monotonic()-started)
        metadata['processed_fps']=counts['frames']/max(metadata['elapsed_s'],1e-9)
        (session/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps({'session':str(session),'status':metadata['status'],'counts':counts}))
    return 130 if metadata['status']=='cancelled' else 0


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (ValueError,RuntimeError,OSError,ImportError) as exc:
        print(f'CO-Sign collection: {exc}',file=sys.stderr)
        raise SystemExit(2)
