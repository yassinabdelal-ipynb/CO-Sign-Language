# Walkthrough

1. Run setup, doctor, tests and the event replay in the README. Replay emits A–Z and all three phrases; it needs no camera or weights.
2. Open `config/language.json`. Pick a letter's three directions. Four-direction codes are reserved for the phrases. No complete three-direction command is a prefix of a four-direction command.
3. Supply a trusted checkpoint. Try `--image` on an already cropped eye. Inspect all six probabilities rather than treating a high score as an accuracy measurement.
4. Supply the dlib predictor or an explicit manual ROI and open webcam mode. Establish center, hold each direction, and return to center between events. Test held directions, temporary face loss and timeout behavior. Predictions do not silently record the camera.
5. For collection, obtain permission and choose one label. Use manual mode only when deliberately declaring the eye state. Inspect `metadata.json` and rejected rows before assembling a dataset. Automatic YOLO mode requires the missing custom weights.
6. Split participants/sessions, train, and evaluate an untouched test set before reporting recognition or communication performance. A corrected image prediction is not equivalent to a correctly typed phrase.

## A 60–90 second recorded demonstration

- **0–15 s:** Show the four main files and run the CPU tests. Explain that this release reconstructs missing runtime pieces around a supplied model.
- **15–30 s:** Run event replay. Show one three-event letter and one four-event phrase in the codebook. State that the inputs are labeled synthetic events.
- **30–50 s:** With local weights available, show the checkpoint image prediction or a consented webcam session. Demonstrate center re-arm and how holding a direction produces one event. Do not claim webcam accuracy from a brief example.
- **50–65 s:** Open the collection diagram and a private sample manifest. Point out manual versus YOLO state evidence; no trained YOLO weights are bundled.
- **65–80 s:** Show the supplied validation curves and the new GPU timing samples. Distinguish original reported 22 words/minute from these inference measurements.
- **80–90 s:** Show the official ISEF award link and the missing-components list. Explain the next evaluation needed: participant-level generalization, false commands and text-entry timing.

The video folder contains instructions only until actual recordings are added. Do not label synthetic replay as a participant test or record private camera images without permission.
