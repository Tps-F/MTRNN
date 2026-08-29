# MTRNN

A PyTorch port of the multiple timescale recurrent neural network from

> Yamashita Y, Tani J (2008). *Emergence of Functional Hierarchy in a Multiple
> Timescale Neural Network Model: A Humanoid Robot Experiment.*
> PLoS Comput Biol 4(11): e1000220.

The point of the model is that no hierarchy is built into the architecture. One
fully connected population of firing-rate units learns everything; the units
differ only in their time constant, and a functional hierarchy — reusable motor
primitives in the fast units, sequences of primitives in the slow ones — falls
out of that difference alone.

## Install

```sh
uv sync            # or: pip install -e ".[dev]"
```

## Three pieces

**`SparseCoder` / `TPM`** (`mtrnn/tpm.py`) — one topology-preserving map (Kohonen
SOM) per sensory modality. `encode` turns a raw frame into a soft activation over
the map's reference vectors; `decode` takes the expectation back to a raw frame
(paper Eqs 2–4). `fit` trains the maps unsupervised, which the authors' C sample
leaves for you to write. Maps are frozen during MTRNN training — their reference
vectors are buffers, so they never reach the optimizer.

**`MTRNN`** (`mtrnn/model.py`) — the continuous-time RNN (Eqs 5–7):

```
u_t = (1 - 1/tau) u_{t-1} + (1/tau) W x_t
```

with `x_t` the sparse input code for the input-output units and the context
units' own previous firing rates for the rest. Each input-output group is a
softmax over itself (so it matches one map's output distribution); context units
use a sigmoid. A fixed `mask` holds the forbidden connections at zero.

**`MTRNNSystem`** — the two closed into a sensori-motor loop (Figure 10). Its
`rollout` returns the full membrane-potential trajectory `(B, T, N)`; firing
rates, predictions and the loss are all pure functions of that one tensor, and
its last `n_slow` columns are what the paper runs PCA on.

## Use

```python
import torch
from mtrnn import MTRNN, MTRNNSystem, Modality, SparseCoder, generate, train

targets = ...  # (n_sequences, n_steps, n_dims) raw sensori-motor recordings

coder = SparseCoder([Modality("proprioception", 8, (8, 8)),
                     Modality("vision", 2, (6, 6))])
coder.fit(targets)                                   # unsupervised, done first
system = MTRNNSystem(coder, MTRNN(io_units=coder.units,
                                  n_fast=60, n_slow=20,
                                  tau_io=2.0, tau_fast=5.0, tau_slow=70.0))

# One binary code per behaviour, in the slow context units: this is the only
# thing that tells the network which sequence to produce.
slow_code = torch.eye(len(targets), 20)[:, :20]
u0 = system.initial_state(targets, slow_code=slow_code)

train(system, targets, u0, closed_rate=0.9, epochs=5000)
prediction, potentials = generate(system, targets, u0, closed_rate=1.0)
```

`closed_rate` is how much of the network's own prediction is fed back instead of
the target (Eq 12): `0.9` during training, `1.0` for free generation, `0.0` for
teacher forcing. Pass one value per modality to differ between them.

To let the initial states self-organize instead of setting them by hand, hand
`train` a `u0` that is an `nn.Parameter` with `requires_grad=True`.

Sequences of different lengths go in right-padded, with a `mask` — see
`mtrnn.data.pad`.

### Additional training of novel sequences

Section *Additional Training of Novel Sequences*: freeze everything but the
connections between the fast and slow context units, and give the new behaviour a
fresh slow-context code.

```python
n_io, n_fast = system.net.n_io, system.net.n_fast
trainable = torch.zeros_like(system.net.mask)
trainable[n_io:n_io + n_fast, n_io + n_fast:] = 1   # slow context -> fast context
trainable[n_io + n_fast:, n_io:n_io + n_fast] = 1   # fast context -> slow context
system.net.weight.register_hook(lambda g: g * trainable)
```

## Running the authors' sample data

The authors' 2011 C sample (`Yamashita Sample Sept 2011`) ships teaching
sequences, trained maps and trained weights. `mtrnn.data.load_reference` reads
that directory format:

```sh
python examples/train_sample.py --reference "path/to/Yamashita Sample Sept 2011/tmp0" --replay
python examples/train_sample.py --reference "path/to/Yamashita Sample Sept 2011/tmp0" --epochs 5000
```

## Is it correct?

`tests/test_matches_c_reference` replays the authors' published weights through
this implementation and compares against trajectories dumped from their C
program. Over 248 fully closed-loop steps the membrane potentials agree to 4e-14
— double-precision round-off, i.e. the two are the same computation — and the
per-sequence scores reproduce what the C program prints:

| sequence | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| closed-loop MSE | .000164 | .000112 | .000146 | .000205 | .000151 | .000193 | .000088 | .000227 | .000240 |

A second trajectory at `delay=2` is checked too: it is what pins down *when* a
prediction comes back as feedback (step `t` -> step `t + delay`), which `delay=1`
cannot distinguish from a plain one-step loop.

`SparseCoder.fit` was checked the same way: on the authors' recordings it reaches
their published maps' quantization error (arm 0.0879 vs 0.0883, vision 0.0356 vs
0.0323).

Training from scratch on the same recordings (9 sequences, 5000 epochs, Adam at
1e-2, ~7 min on a laptop CPU) lands at closed-loop MSE 0.000111 / KL 0.059,
against 0.000169 / 0.083 for the authors' published weights. The loss still throws
an occasional spike late in training — fully closed-loop BPTT over 248 steps does
that — and recovers within a couple of hundred epochs; lower `lr` or `grad_clip`
if you need a smoother curve.

```sh
pytest
```

## Where this differs from the paper

- **Optimizer.** The paper hand-derives BPTT (Eqs 9–11) and runs SGD with
  momentum. Here autograd does the derivation and `train` defaults to Adam;
  pass your own `optimizer` to change that. Gradient norms are clipped by
  default, standing in for the `tanh` squash the C code applies to context
  deltas — long closed-loop rollouts otherwise throw occasional spikes.
- **`elman=True` by default.** The paper fixes at zero only the connections
  between the two modalities. The authors' C code additionally cuts *every*
  input-output → input-output connection (`ELMAN_TYPE_NET`), which is how their
  published weights were produced, so that is the default here. Pass
  `MTRNN(elman=False)` for the literal reading of the paper.
- **No bias.** The paper has none, and the C code allocates one but never updates
  it. `MTRNN(bias=True)` adds a trainable one, inside the integration as in Eq 5;
  the C code would have added it at the activation instead, so the two are not
  interchangeable if you ever turn it on.
- **Not ported:** the `TPMNeighbor` sparse input-connection scheme, which is dead
  code under `ELMAN_TYPE_NET`, and the real-time X11 plotting.
- The `L.G.initA` shipped with the C sample does not quite match
  `MTRNNSystem.initial_state` on its input-output units (mean ~1.4 in log space)
  — it appears to have been generated against slightly earlier map weights. Its
  slow-context half is exactly ±logit(0.95), a 5-bit behaviour code repeated four
  times over the 20 slow units.

## Layout

```
src/mtrnn/tpm.py     topology-preserving maps, sparse coding, SOM training
src/mtrnn/model.py   MTRNN, MTRNNSystem, connection mask
src/mtrnn/train.py   teaching signal, KL objective, train / generate / report
src/mtrnn/data.py    readers for the reference C file formats
examples/            runnable starting point
tests/               correctness checks, incl. the C golden trajectory
```
