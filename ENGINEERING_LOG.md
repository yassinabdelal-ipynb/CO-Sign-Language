# Engineering log

## 2026-10-16 — source inspection

The author supplied a preprocessing script, GPU notebook, poster excerpts and Canva link. Adjacent local webcam scripts and a compatible checkpoint/training artifacts were found. Original files were preserved outside this repository; source hashes and a text copy of the original preprocessing script are retained.

Observed conflicting crop descriptions: whole-frame polygon masks at 100×50 in the supplied script; author-described 350×150 cropped eyes; notebook validation inputs at 224×224. Decision: share new collector/live crop geometry and exactly match the notebook's deterministic grayscale/nearest224/rescale conversion, without claiming historical crop equivalence.

Observed fixed-token ViT code despite historical A-ViT terminology. Preserved checkpoint architecture, class ordering and attention inner width; documented that adaptive halting is absent. Disabled the notebook's direction-changing horizontal flip for future training and added fixed seeds; historical numerical artifacts were not changed or relabeled as new training.

## 2026-10-16 — reconstruction and evidence boundaries

The author confirmed that original command combinations could be replaced, YOLO weights were unavailable, visible poster faces were the team, and 22 words/minute was the desired historical figure. Implemented a seeded, prefix-free codebook with 26 three-event letters and three four-event phrases. Confidence, dwell, center re-arm, camera-stall interruption and command timeout are new debounce rules.

The collector implements the described MediaPipe/dlib/crop/grayscale/optional-noise pipeline. Automatic eye-state filtering needs explicit custom weights; a declared manual state is a separate mode. The closed-eye/blink rule is tested independently. No original intention code or calibrated gaze inputs were supplied, so no intention classifier was fabricated or enabled. Documented Kalman/NIS mathematics and its limitations.

Verified the ISEF fourth award against the official Society for Science list, with both team members credited. Preserved author-provided screenshots. Recorded the poster's 20-versus-author-reported-22 words/minute discrepancy instead of changing the historical chart. The Canva board was not retrievable through the web tool; only supplied excerpts are included.

## 2026-10-16 — verification

Initial CPU suite: 15 passed and one cancellation fixture failed because the mock camera's FPS method returned an unserializable mock object. Corrected the fixture to return 30.0; 16 tests then passed. No production result was invented from this fixture.

Loaded the real supplied checkpoint with `weights_only=True` and strict state loading. Verified CPU/CUDA probabilities on deterministic generated pixels on the actual RTX 5060 Laptop GPU. No camera was opened or participant dataset evaluated.

The first timing attempt warmed only model execution; its request path had a 59.57 ms first sample. Changed warmup to exercise both measured scopes. Retained that initial report and both corrected reports. Final latency/variability and exact environment are in `docs/RESULTS.md`; laptop conditions were uncontrolled and no speedup is claimed.

Rendered Matplotlib training/confusion plots from supplied numerical artifacts and collection/model flow diagrams from this code. Visually inspected the training plot and collection diagram. Parsed all PowerShell scripts, compiled notebook code cells, cleared private notebook outputs, and audited the staged publication set for credentials, private paths, large weights and recordings. Camera GUI and absent YOLO/intention components remain unverified.
