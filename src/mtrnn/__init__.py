"""A PyTorch port of the multiple timescale RNN of Yamashita & Tani (2008).

Yamashita Y, Tani J (2008) Emergence of Functional Hierarchy in a Multiple
Timescale Neural Network Model: A Humanoid Robot Experiment.
PLoS Comput Biol 4(11): e1000220.
"""

from .model import MTRNN, MTRNNSystem, connection_mask
from .tpm import PAPER_MODALITIES, TPM, Modality, SparseCoder
from .train import generate, kl_divergence, report, teacher, train

__all__ = [
    "MTRNN",
    "MTRNNSystem",
    "connection_mask",
    "TPM",
    "SparseCoder",
    "Modality",
    "PAPER_MODALITIES",
    "train",
    "generate",
    "report",
    "teacher",
    "kl_divergence",
]
