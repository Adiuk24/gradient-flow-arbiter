# gradient-flow arbiter

Verification tooling for training a language model on a framework younger than its ecosystem's
collective debugging experience.

Companion artifact to **"Training a Language Model End-to-End in Rust: An Experience Report"**
(Arif Ahmed Adito, Adioris Tech Ltd., Dhaka).

## The problem these tests exist for

The dominant failure mode of training outside the PyTorch/CUDA ecosystem is **not a crash**. It is a
run that compiles, launches, produces a smoothly falling loss curve, and learns nothing where it
matters. Across a from-scratch Rust pretraining project I hit eight such defects across two
frameworks. **Every one of them passed ordinary loss-curve inspection. None announced itself.**

These are the tests that caught them. They are framework-agnostic, cost seconds, and need no
knowledge of *why* something might be broken — only that it must not be.

## Contents

| file | catches |
|---|---|
| `gradient_flow_arbiter.py` | parameters receiving **no gradient at all** — dead branches, detached subgraphs, and fused kernels registered without a backward implementation |
| `tests/gradient_parity.py` | a fused kernel whose backward is wrong, or which is never actually called because the public forward hits the wrong entry point |
| `tests/position_effect_probe.py` | positional encoding that is computed but has **no effect** (e.g. rotary applied on the head axis instead of the sequence axis) |
| `tests/data_order_check.py` | domain-grouped shard order, which makes the model serially forget each domain and sawtooths the loss curve |

Every file is runnable standalone and ships **both** a positive and a negative control — each test
proves it catches the bug it claims to catch, not merely that it passes on healthy input:

```bash
python3 gradient_flow_arbiter.py
python3 tests/gradient_parity.py
python3 tests/position_effect_probe.py
python3 tests/data_order_check.py
```

Only `torch` is required, and only for the bindings — the arbiter's core (`check_gradients`) takes
four callables and has no framework dependency at all. A Candle, Burn, or JAX binding is about
fifteen lines.

## The one I'd most want you to take

`gradient_flow_arbiter.py`. After one forward/backward on a small but structurally complete model,
it asserts every trainable parameter received a gradient that is **present, finite, and not
identically zero**, and fails by parameter name.

That single test caught a framework's fused `layer_norm`, `rms_norm`, `softmax` and `rotary_emb`
silently yielding no gradient — before a paid training run, rather than after one.

I recommend it unconditionally, on any framework, independent of language.

## Two notes on honesty

Writing these, two of my own test files were themselves wrong, and running them is what caught it:
a parity check that reduced with `.sum()` (vacuous for anything summing to a constant — softmax's
true gradient under `.sum()` *is* zero, so a zero-returning backward passes), and an ordering check
whose "balanced" control was 60/40. Both are fixed, and the first is commented in place.
A test you have never seen fail is not evidence.

The paper's training stack and per-step logs are available on request
(`adittoarif@gmail.com`). Model weights are not released — the run predates a tokenizer correction
described in the paper, and the encoder they require is deliberately blocked as defective.

## Licence

MIT. Steal it.
