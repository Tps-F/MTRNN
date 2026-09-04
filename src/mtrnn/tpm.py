"""Topology-preserving maps (Kohonen SOMs) used for sparse population coding.

Yamashita & Tani (2008), Equations 2-4: each sensory channel gets its own map, a
raw frame is turned into a soft activation over that map's reference vectors, and
the network's output distribution is turned back into a raw frame by taking the
distribution's expectation over the same vectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Modality:
    """One sensory channel and the map that encodes it."""

    name: str
    dim: int  # raw sensor dimensions
    shape: tuple[int, int]  # map grid, (rows, cols)
    sigma: float = 0.01  # width of the encoding softmax

    @property
    def units(self) -> int:
        return self.shape[0] * self.shape[1]


#: The humanoid experiment of the paper: 8 arm joints on an 8x8 map, 2 vision dims on 6x6.
PAPER_MODALITIES = (
    Modality("proprioception", 8, (8, 8)),
    Modality("vision", 2, (6, 6)),
)


class TPM(nn.Module):
    """A single topology-preserving map.

    The reference vectors are a buffer rather than a parameter: the paper trains
    the maps unsupervised (:meth:`fit`) *before* the MTRNN and freezes them.
    """

    ref: Tensor
    grid: Tensor

    def __init__(self, modality: Modality) -> None:
        super().__init__()
        self.modality = modality
        self.register_buffer("ref", torch.rand(modality.units, modality.dim))
        rows, cols = modality.shape
        self.register_buffer(
            "grid",
            torch.cartesian_prod(torch.arange(rows), torch.arange(cols)).float(),
        )

    def encode(self, x: Tensor) -> Tensor:
        """(..., dim) raw frame -> (..., units) activation distribution (Eq 3)."""
        return torch.softmax(self._logits(x), -1)

    def log_encode(self, x: Tensor) -> Tensor:
        """Same as :meth:`encode` in log space, without underflowing in float32."""
        return torch.log_softmax(self._logits(x), -1)

    def _logits(self, x: Tensor) -> Tensor:
        return -(x.unsqueeze(-2) - self.ref).pow(2).sum(-1) / self.modality.sigma

    def decode(self, p: Tensor) -> Tensor:
        """(..., units) distribution -> (..., dim) raw frame (Eq 4)."""
        return p @ self.ref

    @torch.no_grad()
    def fit(
        self,
        data: Tensor,
        *,
        steps: int = 3000,
        batch: int = 64,
        lr: tuple[float, float] = (0.9, 0.05),
        radius: float | None = None,
        final_radius: float = 0.5,
        generator: torch.Generator | None = None,
    ) -> "TPM":
        """Train the reference vectors with mini-batch Kohonen learning.

        Batch SOM: each unit is pulled toward the neighbourhood-weighted mean of
        a sampled batch, with the learning rate and the neighbourhood radius decaying
        geometrically from `lr[0]`/`radius` to `lr[1]`/`final_radius`.

        The paper trains its maps on the teaching sequences plus samples from
        slightly outside the task range, so that the encoding stays smooth at the
        edges of the workspace and less is lost in the round trip.
        """
        data = torch.as_tensor(data, dtype=self.ref.dtype, device=self.ref.device)
        data = data.reshape(-1, self.modality.dim)

        def pick(n: int) -> Tensor:
            return data[torch.randint(len(data), (n,), generator=generator)]

        self.ref.copy_(pick(self.modality.units))
        r0 = max(self.modality.shape) / 2 if radius is None else radius
        for s in range(steps):
            t = s / max(steps - 1, 1)
            rate = lr[0] * (lr[1] / lr[0]) ** t
            r = r0 * (final_radius / r0) ** t
            x = pick(batch)
            bmu = (x.unsqueeze(-2) - self.ref).pow(2).sum(-1).argmin(-1)
            near = (self.grid[bmu].unsqueeze(-2) - self.grid).pow(2).sum(-1)
            h = torch.exp(-near / (2 * r * r))  # (batch, units)
            # Each unit moves toward the neighbourhood-weighted mean of the batch;
            # units no sample voted for stay where they are.
            votes = h.sum(0).unsqueeze(-1)
            centre = torch.where(
                votes > 1e-12, (h.T @ x) / votes.clamp_min(1e-12), self.ref
            )
            self.ref.lerp_(centre, rate)
        return self

    def quantization_error(self, data: Tensor) -> Tensor:
        """Mean distance from each sample to its best-matching reference vector."""
        data = data.reshape(-1, self.modality.dim)
        return (
            (data.unsqueeze(-2) - self.ref).pow(2).sum(-1).min(-1).values.sqrt().mean()
        )


class SparseCoder(nn.Module):
    """One TPM per modality, concatenated (the input/output stage of Figure 10)."""

    def __init__(self, modalities: Sequence[Modality] = PAPER_MODALITIES) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        self.maps = nn.ModuleList(TPM(m) for m in self.modalities)
        self.dims = [m.dim for m in self.modalities]
        self.units = [m.units for m in self.modalities]

    def encode(self, x: Tensor) -> Tensor:
        """(..., sum dims) -> (..., sum units)."""
        parts = x.split(self.dims, -1)
        return torch.cat([m.encode(p) for m, p in zip(self.maps, parts)], -1)

    def log_encode(self, x: Tensor) -> Tensor:
        """Same as :meth:`encode` in log space, without underflowing in float32."""
        parts = x.split(self.dims, -1)
        return torch.cat([m.log_encode(p) for m, p in zip(self.maps, parts)], -1)

    def decode(self, p: Tensor) -> Tensor:
        """(..., sum units) -> (..., sum dims)."""
        parts = p.split(self.units, -1)
        return torch.cat([m.decode(q) for m, q in zip(self.maps, parts)], -1)

    def fit(self, data: Tensor, **kwargs) -> "SparseCoder":
        """Train every map on its own slice of `data` (..., sum dims)."""
        parts = torch.as_tensor(data).reshape(-1, sum(self.dims)).split(self.dims, -1)
        for m, part in zip(self.maps, parts):
            m.fit(part, **kwargs)
        return self
