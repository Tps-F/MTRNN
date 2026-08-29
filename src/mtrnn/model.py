"""The multiple timescale recurrent neural network of Yamashita & Tani (2008).

:class:`MTRNN` is the continuous-time RNN itself (Equations 5-7): one fully
connected population whose units differ only in their time constant, plus a fixed
mask of connections the paper forbids. :class:`MTRNNSystem` closes the
sensori-motor loop around it with a :class:`~mtrnn.tpm.SparseCoder` (Figure 10).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .tpm import SparseCoder

#: Slow-context initial states in the published data are +-logit(0.95).
SLOW_LOGIT = 2.944439


def connection_mask(
    io_units: Sequence[int], n_fast: int, n_slow: int, elman: bool = True
) -> Tensor:
    """Which of the N x N connections are allowed to be non-zero.

    The paper fixes two blocks at zero: the two input modalities are not wired to
    each other, and the slow context units are not wired to the input-output units
    (so the slow units can only talk to the world through the fast ones).

    `elman=True` additionally cuts *every* input-output -> input-output
    connection, which is what the authors' reference C code does by default
    (`ELMAN_TYPE_NET`) and what produced their published weights. `elman=False`
    is the literal reading of the paper, cutting only the cross-modality block.
    """
    n_io = sum(io_units)
    n = n_io + n_fast + n_slow
    mask = torch.ones(n, n)
    if elman:
        mask[:n_io, :n_io] = 0
    else:
        group = torch.repeat_interleave(
            torch.arange(len(io_units)), torch.tensor(list(io_units))
        )
        mask[:n_io, :n_io] = (group[:, None] == group[None, :]).float()
    slow = slice(n_io + n_fast, n)
    mask[:n_io, slow] = 0
    mask[slow, :n_io] = 0
    return mask


class MTRNN(nn.Module):
    """Firing-rate CTRNN whose units carry three different time constants.

    Units are laid out as ``[io groups..., fast context, slow context]``. Each
    input-output group is a softmax over itself (matching one TPM's output
    distribution); context units use a sigmoid.

    Defaults are the paper's robot experiment: 64 proprioception + 36 vision
    input-output units (tau 2), 60 fast context units (tau 5), 20 slow context
    units (tau 70), for 180 units in total.
    """

    tau: Tensor
    mask: Tensor

    def __init__(
        self,
        io_units: Sequence[int] = (64, 36),
        n_fast: int = 60,
        n_slow: int = 20,
        tau_io: float = 2.0,
        tau_fast: float = 5.0,
        tau_slow: float = 70.0,
        *,
        elman: bool = True,
        bias: bool = False,
        init_scale: float = 0.025,
    ) -> None:
        super().__init__()
        self.io_units = tuple(io_units)
        self.n_io = sum(self.io_units)
        self.n_fast, self.n_slow = n_fast, n_slow
        self.n_units = self.n_io + n_fast + n_slow

        self.weight = nn.Parameter(
            torch.empty(self.n_units, self.n_units).uniform_(-init_scale, init_scale)
        )
        self.bias = nn.Parameter(torch.zeros(self.n_units)) if bias else None
        # Store tau, not 1/tau: the paper's values (2, 5, 70) are exact in any
        # float, their reciprocals are not, so a float32 buffer widened by
        # `.double()` would keep float32's 0.2 and cap the whole model at ~1e-6.
        self.register_buffer(
            "tau",
            torch.cat(
                [
                    torch.full((self.n_io,), tau_io),
                    torch.full((n_fast,), tau_fast),
                    torch.full((n_slow,), tau_slow),
                ]
            ),
        )
        self.register_buffer(
            "mask", connection_mask(self.io_units, n_fast, n_slow, elman)
        )

    @property
    def n_context(self) -> int:
        return self.n_fast + self.n_slow

    @property
    def alpha(self) -> Tensor:
        """Leak rate 1/tau, at whatever dtype the module currently holds."""
        return 1.0 / self.tau

    def activate(self, u: Tensor) -> Tensor:
        """Membrane potentials -> firing rates (Eq 6)."""
        io, ctx = u.split([self.n_io, self.n_context], -1)
        groups = [g.softmax(-1) for g in io.split(self.io_units, -1)]
        return torch.cat([*groups, ctx.sigmoid()], -1)

    def log_activate_io(self, u: Tensor) -> Tensor:
        """Log firing rates of the input-output units only, computed stably."""
        io = u[..., : self.n_io]
        return torch.cat([g.log_softmax(-1) for g in io.split(self.io_units, -1)], -1)

    def step(self, p: Tensor, u: Tensor) -> Tensor:
        """One integration step (Eqs 5 and 7).

        `p` is the external sparse code for this step (B, n_io) and `u` the
        previous membrane potentials (B, N); the context units feed back their own
        previous firing rates. Returns the new membrane potentials.
        """
        x = torch.cat([p, self.activate(u)[..., self.n_io :]], -1)
        drive = F.linear(x, self.weight * self.mask, self.bias)
        alpha = self.alpha
        return (1 - alpha) * u + alpha * drive


class MTRNNSystem(nn.Module):
    """Sparse coder + MTRNN wired into a closed sensori-motor loop (Figure 10)."""

    def __init__(self, coder: SparseCoder | None = None, net: MTRNN | None = None):
        super().__init__()
        self.coder = coder if coder is not None else SparseCoder()
        self.net = net if net is not None else MTRNN(tuple(self.coder.units))
        if self.net.io_units != tuple(self.coder.units):
            raise ValueError(
                f"input-output groups {self.net.io_units} do not match the maps "
                f"{tuple(self.coder.units)}"
            )

    def predict(self, u: Tensor) -> Tensor:
        """Membrane potentials -> predicted raw sensori-motor state."""
        return self.coder.decode(self.net.activate(u)[..., : self.net.n_io])

    def initial_state(
        self,
        targets: Tensor,
        *,
        delay: int = 1,
        slow_code: Tensor | None = None,
        logit: float = SLOW_LOGIT,
    ) -> Tensor:
        """Initial membrane potentials u_{-1}, one row per sequence.

        Input-output units start from the log of the first desired output, fast
        context units from the neutral value 0, and slow context units from a
        per-sequence binary code -- this is the only thing that distinguishes one
        learned behaviour from another. `slow_code` is (B, n_slow) in {0, 1};
        the default leaves the slow units neutral too, which is only useful when
        the initial states are themselves being learned.
        """
        io = self.coder.log_encode(targets[:, delay])
        ctx = io.new_zeros(io.shape[0], self.net.n_context)
        if slow_code is not None:
            ctx[:, self.net.n_fast :] = (2 * slow_code.to(ctx) - 1) * logit
        return torch.cat([io, ctx], -1)

    def rollout(
        self,
        targets: Tensor,
        u0: Tensor,
        closed_rate: float | Sequence[float] = 1.0,
        delay: int = 1,
    ) -> Tensor:
        """Run the loop over `targets` (B, T, D); return potentials (B, T, N).

        `closed_rate` is how much of the network's own prediction is fed back as
        the next input (Eq 12): 1.0 is fully closed-loop generation, 0.0 is fully
        open-loop teacher forcing, and the paper trains at 0.9. Pass a scalar or
        one value per modality.

        The network predicts `delay` steps ahead, so the prediction made at step
        t is what comes back as feedback at step t + `delay`; the first `delay`
        steps have no prediction yet and see the target.
        """
        rate = self._rate(closed_rate, targets)
        u, out = u0, []
        feedback = list(targets[:, :delay].unbind(1))
        for t in range(targets.shape[1]):
            world = rate * feedback[t] + (1 - rate) * targets[:, t]
            u = self.net.step(self.coder.encode(world), u)
            out.append(u)
            feedback.append(self.predict(u))
        return torch.stack(out, 1)

    def _rate(self, closed_rate: float | Sequence[float], ref: Tensor) -> Tensor:
        r = torch.as_tensor(closed_rate, dtype=ref.dtype, device=ref.device)
        if r.ndim == 0:
            return r
        return r.repeat_interleave(torch.tensor(self.coder.dims, device=r.device))
