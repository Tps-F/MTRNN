# MTRNN

A Python (PyTorch) reimplementation of the multiple timescale recurrent neural
network of Yamashita & Tani (2008), [PLoS Comput Biol 4(11):
e1000220](https://doi.org/10.1371/journal.pcbi.1000220). The paper is the
specification. The original work came out of Jun Tani's lab, now the [Cognitive
Neurorobotics Research Unit at
OIST](https://www.oist.jp/research/research-units/cnru/past-projects).

No hierarchy is built into the architecture. One fully connected population of
firing-rate units learns everything, and the units differ only in their time
constant. The functional hierarchy the paper is about, motor primitives in the
fast units and sequences of primitives in the slow ones, comes out of that
difference alone.

The authors released a C sample in 2011 along with trained weights. It trains
the network but not the Kohonen maps that encode the network's input: "if you
want to train a network by using your original data, you have to develop another
codes to train a Kohonen map", as its README puts it, and `Kohonen.c` indeed only
reads, encodes and decodes. That missing step is `SparseCoder.fit` here, so this
code trains end to end on your own recordings.

Replaying their published weights through this code reproduces a trajectory
dumped from their C program to 4e-14 over 248 closed-loop steps, which is what
`pytest` checks.

```sh
uv sync            # or: pip install -e ".[dev]"
pytest
```

## Use

```python
import numpy as np, torch
from mtrnn import MTRNN, MTRNNSystem, Modality, SparseCoder, generate, train

# (n_sequences, n_steps, n_dims): 8 arm joints then 2 object-position dims
targets = torch.from_numpy(np.load("recordings.npy")).double()

coder = SparseCoder([Modality("proprioception", 8, (8, 8)),
                     Modality("vision", 2, (6, 6))])
coder.fit(targets)

# elman=True (the default) also cuts the within-modality input-output
# connections the paper permits, matching the authors' released C code and the
# weights it produced. Pass elman=False for the paper's wiring.
system = MTRNNSystem(coder, MTRNN(io_units=coder.units, n_fast=60, n_slow=20,
                                  tau_io=2.0, tau_fast=5.0, tau_slow=70.0)).double()

# One binary code per behaviour in the slow context units. This is the only thing
# that tells the network which sequence to produce.
u0 = system.initial_state(targets, slow_code=torch.eye(len(targets), 20))

train(system, targets, u0, closed_rate=0.9, epochs=5000)
prediction, potentials = generate(system, targets, u0, closed_rate=1.0)
```

`closed_rate` is how much of the network's own prediction is fed back instead of
the target (Eq 12): 0.9 while training, 1.0 for free generation, 0.0 for teacher
forcing. One value per modality also works. `delay` is how far ahead the network
predicts, so the output of step `t` returns as feedback at step `t + delay`. A
`u0` that is an `nn.Parameter` joins the optimizer and learns. Ragged sequences
go in right-padded with a `mask` from `mtrnn.data.pad`. `train` uses Adam and
clips gradients where the paper hand-derives BPTT and runs SGD with momentum;
pass `optimizer=` to change that.

## Sample data

`python examples/train_sample.py` takes no arguments. It fits the maps, derives
the initial states and learns the weights from the recordings committed as this
repository's test fixture. `--replay` evaluates the authors' published weights
instead of training.

One from-scratch run (5000 epochs, seed 0) reaches mean closed-loop MSE 0.000169
and KL 0.072, against 0.000169 and 0.083 for the published weights. It is less
even across behaviours than they are, though: its worst sequence scores 0.000492
where theirs scores 0.000240, and its best 0.000039 where theirs scores 0.000088.

## Citation

```bibtex
@article{yamashita2008mtrnn,
  author  = {Yamashita, Yuichi and Tani, Jun},
  title   = {Emergence of Functional Hierarchy in a Multiple Timescale
             Neural Network Model: A Humanoid Robot Experiment},
  journal = {PLoS Computational Biology},
  volume  = {4},
  number  = {11},
  pages   = {e1000220},
  year    = {2008},
  doi     = {10.1371/journal.pcbi.1000220}
}
```
