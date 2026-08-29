"""Readers for the file formats of the authors' reference C implementation.

Useful for validating this port against the published weights, and as a template
for loading your own recordings. Nothing else in the package depends on them.

Every reader returns float64: the reference files carry 16 significant digits and
reading them at float32 is what would otherwise limit the agreement with the C
program. :func:`load_reference` casts down at the end if you ask it to.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from .model import MTRNNSystem
from .tpm import SparseCoder


def read_patterns(path: str | Path, dim: int) -> list[Tensor]:
    """Read ``tmp.train.pat``: ``step value...`` rows, sequences split by ``-1``."""
    seqs: list[Tensor] = []
    frames: list[list[float]] = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if float(fields[0]) < 0:
            if frames:
                seqs.append(torch.tensor(frames, dtype=torch.float64))
            frames = []
        else:
            frames.append([float(v) for v in fields[1 : dim + 1]])
    if frames:
        seqs.append(torch.tensor(frames, dtype=torch.float64))
    return seqs


def read_map_weights(path: str | Path, coder: SparseCoder) -> SparseCoder:
    """Read ``tmp.koh.wt`` into `coder`: per map, a unit count then its vectors."""
    tokens = iter(Path(path).read_text().split())
    for tpm in coder.maps:
        n = int(next(tokens))
        if n != tpm.modality.units:
            raise ValueError(
                f"{tpm.modality.name}: file has {n} units, map has {tpm.modality.units}"
            )
        flat = [float(next(tokens)) for _ in range(n * tpm.modality.dim)]
        tpm.ref.copy_(torch.tensor(flat, dtype=torch.float64).view(n, tpm.modality.dim))
    return coder


def read_initial_states(path: str | Path, n_units: int) -> Tensor:
    """Read ``L.G.initA``: a sequence count then one row of potentials per sequence."""
    tokens = Path(path).read_text().split()
    n = int(tokens[0])
    return torch.tensor(
        [float(v) for v in tokens[1 : 1 + n * n_units]], dtype=torch.float64
    ).view(n, n_units)


def read_weights(path: str | Path, n_units: int) -> tuple[Tensor, Tensor]:
    """Read ``tmpL.wtA``: one row per unit, N weights then a bias."""
    flat = torch.tensor(
        [float(v) for v in Path(path).read_text().split()], dtype=torch.float64
    )
    rows = flat.view(n_units, n_units + 1)
    return rows[:, :n_units], rows[:, n_units]


def pad(seqs: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Right-pad variable-length sequences by holding the last frame.

    Returns (B, T, D) targets and a (B, T) float mask of the real steps.
    """
    length = max(len(s) for s in seqs)
    targets = torch.stack(
        [torch.cat([s, s[-1:].expand(length - len(s), -1)]) for s in seqs]
    )
    mask = torch.stack([torch.arange(length) < len(s) for s in seqs]).to(targets.dtype)
    return targets, mask


def load_reference(
    directory: str | Path,
    *,
    weights: str | None = "tmpL.wtA",
    dtype: torch.dtype = torch.float64,
) -> tuple[MTRNNSystem, Tensor, Tensor, Tensor]:
    """Load a ``tmpN/`` directory of the reference implementation.

    Returns ``(system, targets, mask, u0)``. Pass ``weights=None`` to keep random
    MTRNN weights, e.g. to retrain on the published data from scratch.
    """
    d = Path(directory)
    system = MTRNNSystem().to(dtype)
    read_map_weights(d / "tmp.koh.wt", system.coder)
    targets, mask = pad(read_patterns(d / "tmp.train.pat", sum(system.coder.dims)))
    u0 = read_initial_states(d / "L.G.initA", system.net.n_units)
    if weights is not None:
        w, b = read_weights(d / weights, system.net.n_units)
        system.net.weight.data.copy_(w)
        if system.net.bias is None and b.abs().max() > 0:
            raise ValueError(f"{weights} has non-zero biases; build MTRNN(bias=True)")
        elif system.net.bias is not None:
            system.net.bias.data.copy_(b)
    return system, targets.to(dtype), mask.to(dtype), u0.to(dtype)
