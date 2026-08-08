#!/usr/bin/env python
"""Thin shim so `python scripts/train_resnet.py` works. Logic lives in
convolutions.cli (also exposed as the `train-resnet` console command)."""

from convolutions.cli import main

if __name__ == "__main__":
    main()
