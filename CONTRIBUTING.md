# Contributing

Run `python -m unittest discover -s tests -v` in a configured CPU environment. Keep camera/GPU tests optional and report hardware honestly. Reproduce figures with `python scripts/make_figures.py` after installing the training/plot dependencies.

Changes to cropping, interpolation, normalization, class order or command timing need an explicit compatibility note and tests. Explain how a new metric was measured; preserve raw observations and failed attempts. Do not relabel historical figures as new results.

Never commit datasets, participant records, recordings, credentials, local paths, dlib predictors or model weights. Share only authorized, reviewed derivatives. Report security-sensitive material privately to the repository owner rather than including credentials/images in public issues.
