from __future__ import annotations

import torch


def best_device(allow_mps: bool = True) -> str:
    """CUDA on Colab/NVIDIA, Apple-silicon GPU (MPS) on M-series Macs, otherwise CPU.

    Small networks such as the MADDPG actor/critic train faster on CPU than on MPS, so callers can
    opt out of MPS with ``allow_mps=False``.
    """
    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
