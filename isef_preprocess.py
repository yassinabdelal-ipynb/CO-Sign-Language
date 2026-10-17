"""Shared eye-image contract. No camera, model download or file write on import.

The supplied original script masked whole frames at 100x50. The tight 350x150
collector here is a reconstruction of the author's description, not that script.
"""
from pathlib import Path
import numpy as np
from PIL import Image

COLLECT_SIZE = (350, 150)  # width, height
PROFILE = 'rect350x150-pilL-nearest224-v1'


def training_array(image, size=224):
    """Match notebook validation load_img(grayscale, nearest), rescale 1/255.

    Input: PIL image or RGB/grayscale uint8 array; never pass an OpenCV BGR
    array directly. Output: float32 NCHW, shape (1, 1, size, size).
    No random augmentation during inference or validation.
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError('size must be a positive integer')
    if not isinstance(image, Image.Image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.size == 0:
            raise ValueError('image must be a nonempty uint8 RGB/grayscale array')
        image = Image.fromarray(image)
    pixels = np.asarray(image.convert('L').resize((size, size), Image.Resampling.NEAREST))
    return (pixels.astype(np.float32) * np.float32(1 / 255))[None, None]


def roi_crop(frame_bgr, roi):
    """Crop bounded (x,y,width,height), return 350x150 grayscale PIL image."""
    if len(roi) != 4 or not np.isfinite(roi).all():
        raise ValueError('ROI must contain four finite numbers')
    x, y, w, h = map(int, roi)
    if w <= 0 or h <= 0:
        raise ValueError('ROI width/height must be positive')
    height, width = frame_bgr.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(width, x+w), min(height, y+h)
    if x0 >= x1 or y0 >= y1:
        raise ValueError('ROI does not intersect the image')
    rgb = frame_bgr[y0:y1, x0:x1, ::-1]
    # Explicit intermediate collection size. PIL nearest is recorded in PROFILE.
    return Image.fromarray(rgb).convert('L').resize(COLLECT_SIZE, Image.Resampling.NEAREST)


def collect_eye(frame_bgr, points):
    """Dlib six-point eye contour -> padded rectangular ROI, not polygon masking."""
    points = np.asarray(points, dtype=float)
    if points.shape != (6, 2) or not np.isfinite(points).all():
        raise ValueError('Expected six finite (x,y) eye landmarks')
    cx, cy = (points.min(axis=0) + points.max(axis=0)) / 2
    width = max(12., float(np.ptp(points[:, 0])) * 1.5)
    height = width * 150 / 350
    return roi_crop(frame_bgr, (cx-width/2, cy-height/2, width, height))


def augment(image, rng):
    """Collection-only Gaussian noise. Keep originals; split sessions first.

    This is a new optional augmentation, not recovered historical settings.
    Horizontal flipping is intentionally excluded to preserve direction labels.
    """
    pixels = np.asarray(image.convert('L'), dtype=np.float32)
    pixels = np.clip(pixels + rng.normal(0, 3, pixels.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def keep_eye(label, state):
    """Reject unknown gates; retain closed eyes only in the explicit blink class."""
    if label not in {'up', 'down', 'left', 'right', 'center', 'blink'}:
        raise ValueError('Unknown collection label')
    if state not in {'open', 'closed', 'unknown'}:
        raise ValueError('Unknown eye state')
    return state == 'open' or (state == 'closed' and label == 'blink')


class EyeLocator:
    """MediaPipe face detection followed by dlib 68-point facial landmarks.

    Uses pinned legacy MediaPipe 0.10.21. Does not download a dlib predictor.
    Eye selection is image-left or image-right, not anatomical left/right.
    """
    def __init__(self, predictor):
        if not Path(predictor).is_file():
            raise FileNotFoundError('Supply --predictor with a licensed dlib 68-point .dat file')
        import mediapipe as mp
        import dlib
        if not hasattr(mp, 'solutions'):
            raise RuntimeError('This locator requires mediapipe==0.10.21; see requirements-collection.txt')
        self.face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=.6)
        self.predictor = dlib.shape_predictor(str(predictor))
        self.dlib = dlib

    def locate(self, frame_bgr, side='image-left'):
        import cv2
        detections = self.face.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).detections
        if not detections:
            return None
        b = max(detections, key=lambda d: d.score[0]).location_data.relative_bounding_box
        h, w = frame_bgr.shape[:2]
        x0, y0 = max(0, int(b.xmin*w)), max(0, int(b.ymin*h))
        x1, y1 = min(w-1, int((b.xmin+b.width)*w)), min(h-1, int((b.ymin+b.height)*h))
        if x1 <= x0 or y1 <= y0:
            return None
        shape = self.predictor(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), self.dlib.rectangle(x0,y0,x1,y1))
        eyes = [np.array([(shape.part(i).x, shape.part(i).y) for i in range(a,a+6)]) for a in (36,42)]
        eyes.sort(key=lambda p: p[:,0].mean())
        return eyes[0 if side == 'image-left' else 1]

    def close(self):
        self.face.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Convert an already cropped eye to the exact model input.')
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with Image.open(args.input) as image:
        a = training_array(image)
    Image.fromarray(np.rint(a[0,0]*255).astype(np.uint8)).save(args.output)
