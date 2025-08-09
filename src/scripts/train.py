#!/usr/bin/env python3
"""Convenience wrapper for training via src.main CLI."""

import subprocess
import sys

def main():
    cmd = [sys.executable, "-m", "src.main", "--train", "--model-type", "cnn", "--feature-type", "mel_spectrogram"]
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()


