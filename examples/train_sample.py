"""Train the MTRNN from scratch on the authors' sample recordings.

    python examples/train_sample.py

Fits the Kohonen maps, derives the initial states and learns the weights, all
from the recordings committed as this repository's test fixture. The authors' C
sample cannot train the maps -- its README says so outright -- which is the part
of the pipeline this port exists to supply.

    --replay   evaluate the authors' published weights instead of training.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from mtrnn import MTRNNSystem, report, train

FIXTURE = Path(__file__).resolve().parent.parent / "tests/reference/c_reference.npz"


def behaviour_codes(n_seq: int, n_slow: int, bits: int = 5) -> torch.Tensor:
    """One binary code per sequence, tiled across the slow context units.

    The shipped `L.G.initA` uses a 5-bit code repeated four times over its 20
    slow units, which is what distinguishes one learned behaviour from another.
    """
    if n_slow % bits or n_seq > 2**bits:
        raise ValueError(f"{n_seq} sequences do not fit {bits} bits x {n_slow} units")
    code = (torch.arange(n_seq)[:, None] >> torch.arange(bits)) & 1
    return code.repeat(1, n_slow // bits).double()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", action="store_true", help="evaluate published weights")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    ref = np.load(FIXTURE)
    targets = torch.from_numpy(ref["targets"])
    lengths = torch.from_numpy(ref["lengths"])
    mask = (torch.arange(targets.shape[1])[None] < lengths[:, None]).to(targets)
    system = MTRNNSystem().double()

    if args.replay:
        for tpm, name in zip(system.coder.maps, ["proprioception", "vision"]):
            tpm.ref.copy_(torch.from_numpy(ref[f"map_ref_{name}"]))
        system.net.weight.data.copy_(torch.from_numpy(ref["weight"]))
        u0 = torch.from_numpy(ref["u0"])
    else:
        system.coder.fit(torch.cat([t[m.bool()] for t, m in zip(targets, mask)]))
        u0 = system.initial_state(
            targets, slow_code=behaviour_codes(len(targets), system.net.n_slow)
        )
        train(
            system,
            targets,
            u0,
            mask=mask,
            epochs=args.epochs,
            on_epoch=lambda e, kl: e % 50 == 0 and print(f"epoch {e:5d}  KL {kl:.5f}"),
        )

    scores = report(system, targets, u0, mask=mask, closed_rate=1.0)
    for i, (mse, kl) in enumerate(zip(scores["mse"], scores["kl"])):
        print(f"seq {i}: closed-loop MSE {mse:.6f}  KL {kl:.6f}")
    print(f"mean: MSE {scores['mse'].mean():.6f}  KL {scores['kl'].mean():.6f}")


if __name__ == "__main__":
    main()
