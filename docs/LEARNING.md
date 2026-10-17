# Learning notes

## How the code works

**Preprocessing is a contract.** `training_array` uses Pillow grayscale conversion, nearest-neighbor 224×224 resizing and float32 multiplication by 1/255. The order and interpolation matter: visually similar inputs can produce different predictions. The collector and live locator share their new 350×150 geometry; this does not establish the unknown historical training crop.

**Vision Transformer.** `Patches` cuts an input into 7×7 patches, preserving the original TensorFlow patch ordering. `PatchEncoder` projects 49 pixels to width 64 and adds learned position embeddings. Twelve blocks combine attention and a feed-forward residual path, followed by average pooling and a classifier. `KerasStyleAttention` preserves the notebook's per-head key dimension: eight times 64, not 64 divided across eight heads. This architecture is fixed-token; “adaptive ViT” would require additional implemented behavior.

**Decoder state machine.** A confident direction must persist for a dwell interval. A confident center interval re-arms the detector. Prefix-free commands can be emitted immediately once complete, even though some have three events and others four. Repeated frames of the same gaze do not count as repeated commands. Missing/uncertain observations and camera stalls cannot count as a continuous dwell. Timeout clears incomplete input.

**Eye-state filtering.** A closed eye is retained only when collecting `blink`. This rule is independently testable, but applying it automatically requires a trained eye-state classifier. Manual mode is a label declaration, not evidence that the eye was open. The model's `close` class describes an image state; blink detection additionally needs a temporal definition.

**Measurement.** The checkpoint check uses generated inputs to test finite output and CPU/CUDA agreement. GPU work is asynchronous, so CUDA events and synchronization are needed for device execution time. Request timing separately includes preprocessing, transfers and reading results. Neither timing includes the webcam or face/eye detector. Two scopes are warmed independently, all raw timings are retained and laptop conditions can still vary.

**Validation versus independent testing.** The notebook uses its validation split for model selection. Its recorded 99.04% therefore describes validation. A convincing final test needs untouched participants/sessions and separate text-entry measurements, not another visualization of selected examples.

## Five interview questions

1. **Why not just type a letter on every prediction?** Camera frames are correlated. The decoder requires dwell and center re-arm, then consumes three/four events. Otherwise one held gaze could generate many symbols.
2. **How do you decode both three- and four-event commands?** `read_mapping` rejects a complete command that prefixes another. This avoids emitting a three-event letter before receiving a four-event phrase.
3. **What does 99.04% establish?** The supplied validation history and confusion counts agree on that value. The dataset was not rerun during publication and that split selected the checkpoint. It does not establish participant-independent accuracy or 22 words/minute.
4. **Does NIS 5.99 prove intention?** No. Under a calibrated two-dimensional Gaussian innovation model, it is a chi-square consistency threshold. The original intent code/data are missing, so no such filter is active here.
5. **What did you implement versus reuse?** The publication adds shared preprocessing, collection orchestration, a configurable decoder, error handling, manifests and tests around supplied model/notebook code. PyTorch supplies tensor operations/attention; MediaPipe and dlib supply face/landmark detection; optional Ultralytics supplies model inference, not missing trained eye-state weights. Pillow/NumPy handle pixels and Matplotlib renders supplied numerical evidence.

## Exercises

- Change the mapping while preserving 26 letters, three phrases, lengths 3/4 and prefix freedom. Run the validation tests; regenerate a correctly timed event fixture if the codes change.
- Compare `nearest` and bilinear resizing on the same saved crop. Report pixel/output differences before changing the checkpoint input contract.
- Add tests for boundary confidence, face loss mid-command and fast direction changes; measure false command rate on consented labeled streams.
- Train an eye-state classifier on a participant-separated dataset. Evaluate closed-eye rejection and blink retention separately before enabling the automatic gate.
- Recover the original intention code and evaluate it on labeled gaze streams. Preserve its provenance; do not replace missing calibration with guessed coordinates.

## Resume wording

Template: “Co-developed CO-Sign Language, a webcam-based eye-command communication prototype; implemented [your actual components], evaluated on [documented split/participants], achieving [measured metric] under [protocol].”

Repository-verifiable example, only if these changes are reviewed and understood: “Extended an eye-command prototype with a configurable 29-command decoder, shared preprocessing and 16 CPU tests; verified a supplied 1.88M-parameter ViT on an RTX 5060 Laptop GPU with CPU/CUDA probability agreement within 1.79×10⁻⁷ on a generated input.”

Use shared authorship credit and identify later reconstructed code. Original-study accuracy, participants and communication speed need original records; this repository's synthetic tests cannot fill those fields.
