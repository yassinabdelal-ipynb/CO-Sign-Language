# Installation and model assets

Core CPU tests and event replay require PyTorch, NumPy and Pillow. Setup supports Python 3.10–3.12 to keep the optional collection/training packages compatible. Python 3.14 can run the core with newer NumPy/Pillow, as recorded in the local verification, but is outside the pinned collection setup.

Direct dependencies are pinned. Setup writes all resolved transitive versions to the private `local/resolved-requirements.txt`; a universal cross-platform lock is not claimed. Optional collection/training environments have not been freshly installed as part of publication. See `RESULTS.md` for exactly which environments were exercised.

Official references checked during implementation:

- [PyTorch installation selector](https://pytorch.org/get-started/locally/) and [2.14.1 package metadata](https://pypi.org/project/torch/2.14.1/): stable CPU/CUDA distribution, Python requirements. CUDA setup explicitly uses `https://download.pytorch.org/whl/cu130` for RTX 50-series support; doctor and a real inference check establish whether a local installation works.
- [MediaPipe 0.10.21 package](https://pypi.org/project/mediapipe/0.10.21/): deliberately pinned for the legacy `mp.solutions.face_detection` API. Newer MediaPipe Tasks are a different API; upgrading requires code changes and testing.
- [dlib facial-landmark example](https://dlib.net/face_landmark_detection.py.html): predictor requirements and the iBUG 300-W training-data license caveat.
- [Ultralytics prediction API](https://docs.ultralytics.com/modes/predict/): local custom weights, classification probabilities or detection boxes. Version 8.4.178 was verified in official PyPI release metadata; its eye-state adapter remains unverified without a checkpoint.

## Required assets for optional modes

| Asset | Used by | Distribution |
| --- | --- | --- |
| Notebook-produced `comeback_best.pt` | Image/webcam predictions | Supplied locally for testing; not bundled. Train with the notebook or obtain an authorized checkpoint |
| `shape_predictor_68_face_landmarks.dat` | dlib eye localization | Supply a licensed local file; do not assume its training-data license permits every use |
| Custom YOLO `eye_state.pt` with `open` and `closed` classes | Automatic collection filtering | Missing. Must be recovered or trained; generic COCO models are rejected |

The pinned MediaPipe legacy package includes its own pretrained face detector. No separate project download command is needed for that detector. The scripts do not fetch checkpoints or datasets. The setup package installer needs network access; ordinary event replay and local inference use local inputs/assets.

Only load trusted checkpoint files. `torch.load(..., weights_only=True)` is used for the ViT and strict parameter loading rejects mismatches. Optional Ultralytics loading follows that library's model loader; do not give it untrusted weights.

On Windows, a dlib source installation may require Microsoft C++ Build Tools and CMake. This is an explicit manual step if no suitable binary is available; setup will not install system build tools. A build failure leaves the already installed core usable. Installing both `opencv-python` and `opencv-contrib-python` into the same environment can conflict because they provide the same `cv2` module; this project requests the contrib distribution for MediaPipe compatibility. Review the resolved package list if optional Ultralytics dependencies pull in a second OpenCV distribution.

No weights or images are silently replaced with random data. The generated replay stream and generated checkpoint-test input are explicitly identified as synthetic fixtures.
