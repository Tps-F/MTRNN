"""Training and generation for the MTRNN.

The paper derives BPTT by hand (Equations 9-11); here autograd does that, so the
only things worth writing down are the teaching signal, the Kullback-Leibler
objective (Equation 8) and the loop that mixes prediction with target during
learning (Equation 12).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import MTRNN, MTRNNSystem
from .tpm import SparseCoder


def teacher(coder: SparseCoder, targets: Tensor, delay: int = 1) -> Tensor:
    """Desired output distributions y*: the target `delay` steps ahead.

    The last frame is held for the final `delay` steps, as in the reference code.
    """
    tail = targets[:, -1:].expand(-1, delay, -1)
    return coder.encode(torch.cat([targets[:, delay:], tail], 1))


def kl_divergence(net: MTRNN, u: Tensor, teach: Tensor) -> Tensor:
    """Per-step KL divergence from teacher to output, summed over units (Eq 8).

    Returns (B, T). Summing across the input-output groups is the same as adding
    up one divergence per modality, since the groups are separate distributions.

    `xlogy` is what keeps this exact: the teacher is a softmax at sigma=0.01, so
    most of its entries are far below any epsilon one would clamp with, and
    clamping inflates the entropy term. `xlogy(0, 0)` is 0, which is the limit.
    """
    log_y = net.log_activate_io(u)
    return (torch.xlogy(teach, teach) - teach * log_y).sum(-1)


def masked_mean(value: Tensor, mask: Tensor | None) -> Tensor:
    """Mean of `value` (B, T) over real steps only."""
    if mask is None:
        return value.mean()
    return (value * mask).sum() / mask.sum()


def train(
    system: MTRNNSystem,
    targets: Tensor,
    u0: Tensor,
    *,
    mask: Tensor | None = None,
    delay: int = 1,
    closed_rate: float | Sequence[float] = 0.9,
    epochs: int = 5000,
    lr: float = 1e-2,
    grad_clip: float | None = 1.0,
    optimizer: torch.optim.Optimizer | None = None,
    on_epoch: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Fit the network to `targets` (B, T, D) by BPTT; return the loss history.

    Every sequence is rolled out in full each epoch (the sequences *are* the
    batch), which is what the paper does. The sparse coder is frozen -- its
    reference vectors are buffers, so it never enters the optimizer.

    Pass `u0` as an ``nn.Parameter`` with ``requires_grad=True`` to let the
    initial states self-organize instead of being set by hand.

    `grad_clip` bounds the total gradient norm. Backpropagating a fully
    closed-loop rollout over hundreds of steps produces occasional huge
    gradients; the authors' C code squashes them with `tanh`, and clipping is the
    modern equivalent. Set it to None to turn that off.
    """
    params = [p for p in system.parameters() if p.requires_grad]
    if u0.requires_grad:
        params.append(u0)
    opt = optimizer if optimizer is not None else torch.optim.Adam(params, lr=lr)
    teach = teacher(system.coder, targets, delay).detach()

    history = []
    for epoch in range(epochs):
        u = system.rollout(targets, u0, closed_rate, delay)
        loss = masked_mean(kl_divergence(system.net, u, teach), mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        history.append(loss.item())
        if on_epoch is not None:
            on_epoch(epoch, history[-1])
    return history


@torch.no_grad()
def generate(
    system: MTRNNSystem,
    targets: Tensor,
    u0: Tensor,
    closed_rate: float | Sequence[float] = 1.0,
    delay: int = 1,
) -> tuple[Tensor, Tensor]:
    """Closed-loop generation. Returns (predicted sensori-motor states, potentials).

    `targets` still has to be passed: it fixes the number of steps and, when
    `closed_rate < 1`, supplies the sensory feedback the network does not predict
    itself. The returned potentials (B, T, N) are what to run PCA on -- slice
    ``[..., -net.n_slow:]`` for the slow context units.
    """
    u = system.rollout(targets, u0, closed_rate, delay)
    return system.predict(u), u


@torch.no_grad()
def report(
    system: MTRNNSystem,
    targets: Tensor,
    u0: Tensor,
    *,
    mask: Tensor | None = None,
    delay: int = 1,
    closed_rate: float | Sequence[float] = 1.0,
) -> dict[str, Tensor]:
    """Per-sequence MSE over map activations and KL divergence, as the paper reports.

    `mask` marks the real steps of each sequence; the last `delay - 1` of those
    are dropped here, because `teacher` has no target of its own that far ahead
    and holds the final frame instead. That leaves `steps - delay + 1` scored
    steps per sequence, which is the window the reference implementation averages
    over.
    """
    u = system.rollout(targets, u0, closed_rate, delay)
    teach = teacher(system.coder, targets, delay)
    y = system.net.activate(u)[..., : system.net.n_io]
    w = mask if mask is not None else torch.ones(targets.shape[:2], device=u.device)
    w = F.pad(w[:, delay - 1 :], (0, delay - 1))
    per_step_mse = (y - teach).pow(2).mean(-1)
    per_step_kl = kl_divergence(system.net, u, teach)
    return {
        "mse": (per_step_mse * w).sum(1) / w.sum(1),
        "kl": (per_step_kl * w).sum(1) / w.sum(1),
    }
