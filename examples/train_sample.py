"""Train and replay the authors' published sample data.

    python examples/train_sample.py --reference "path/to/Yamashita Sample Sept 2011/tmp0"

With `--replay` it loads their trained weights instead of training, which is the
quickest way to see the model reproduce Figure 3 of the paper.
"""

import argparse
from pathlib import Path

import torch

from mtrnn.data import load_reference
from mtrnn.train import generate, report, train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True, help="a tmpN/ directory")
    p.add_argument("--replay", action="store_true", help="use the published weights")
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--closed-rate", type=float, default=0.9)
    p.add_argument("--out", type=Path, default=Path("runs/sample.pt"))
    args = p.parse_args()

    system, targets, mask, u0 = load_reference(
        args.reference, weights="tmpL.wtA" if args.replay else None
    )
    print(
        f"{targets.shape[0]} sequences, up to {targets.shape[1]} steps, "
        f"{system.net.n_units} units"
    )

    if not args.replay:
        log = lambda e, loss: e % 50 == 0 and print(f"epoch {e:5d}  KL {loss:.5f}")
        train(
            system,
            targets,
            u0,
            mask=mask,
            closed_rate=args.closed_rate,
            epochs=args.epochs,
            lr=args.lr,
            on_epoch=log,
        )

    scores = report(system, targets, u0, mask=mask, closed_rate=1.0)
    for i, (mse, kl) in enumerate(zip(scores["mse"], scores["kl"])):
        print(f"seq {i}: closed-loop MSE {mse:.6f}  KL {kl:.6f}")

    pred, u = generate(system, targets, u0, closed_rate=1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": system.state_dict(),
            "u0": u0,
            "prediction": pred,
            "potentials": u,
            "mask": mask,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
