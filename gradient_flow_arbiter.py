"""
gradient-flow arbiter — the single highest-value test for a training stack built on a
framework younger than its ecosystem's collective debugging experience.

After ONE forward/backward pass on a small but structurally complete model, assert that
every trainable parameter received a gradient that is present, finite, and not identically
zero. Fail loudly, by parameter name.

Why this exists: fused kernels can be registered without a backward implementation. Calling
one inside a training graph then yields NO gradient for the parameters feeding it -- with no
error, no warning, and a loss curve that falls smoothly anyway. See the paper, defect D1.
This test costs seconds and requires no knowledge of WHY a gradient might be missing, only
that it must not be.

Framework-agnostic by design: it needs only (a) an iterator over named trainable parameters
and (b) a way to read each one's gradient. The torch binding below is ~15 lines; a Candle,
Burn, or JAX binding is comparable.

MIT licensed. Steal it.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ArbiterReport:
    checked: int = 0
    failures: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.checked > 0 and not self.failures

    def __str__(self) -> str:
        head = f"gradient-flow arbiter: checked {self.checked} params, {len(self.failures)} failures"
        if self.ok:
            return head + " -> PASS"
        body = "\n".join(f"    {n}: {why}" for n, why in self.failures[:40])
        more = "" if len(self.failures) <= 40 else f"\n    ... and {len(self.failures)-40} more"
        return head + " -> FAIL\n" + body + more


def check_gradients(named_params, get_grad, is_finite, norm) -> ArbiterReport:
    """Core arbiter. Backend-independent.

    named_params : iterable of (name, param) for TRAINABLE params only
    get_grad     : param   -> gradient object or None
    is_finite    : grad    -> bool
    norm         : grad    -> float
    """
    rep = ArbiterReport()
    for name, p in named_params:
        rep.checked += 1
        g = get_grad(p)
        if g is None:
            rep.failures.append((name, "grad is None -- parameter is not in the autograd graph "
                                       "(dead branch, detached subgraph, or a fused op with no backward)"))
            continue
        if not is_finite(g):
            rep.failures.append((name, "grad contains NaN or Inf"))
            continue
        n = norm(g)
        if n == 0.0:
            rep.failures.append((name, "grad is exactly zero -- reachable but receiving no signal"))
    return rep


# ---------------------------------------------------------------- torch binding
def check_torch_model(model, loss) -> ArbiterReport:
    """Run AFTER loss.backward(). Trainable params only."""
    import torch
    return check_gradients(
        named_params=[(n, p) for n, p in model.named_parameters() if p.requires_grad],
        get_grad=lambda p: p.grad,
        is_finite=lambda g: bool(torch.isfinite(g).all()),
        norm=lambda g: float(g.norm()),
    )


def assert_gradient_flow(model, loss) -> ArbiterReport:
    rep = check_torch_model(model, loss)
    if not rep.ok:
        raise AssertionError(str(rep))
    return rep


if __name__ == "__main__":
    # self-check: a model with a deliberately detached branch MUST fail
    import torch, torch.nn as nn

    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.live = nn.Linear(4, 4)
            self.dead = nn.Linear(4, 4)   # contributes via a DETACHED path
        def forward(self, x):
            return self.live(x) + self.dead(x).detach()

    m = Broken()
    loss = m(torch.randn(2, 4)).square().mean()
    loss.backward()
    rep = check_torch_model(m, loss)
    print(rep)
    assert not rep.ok, "arbiter failed to catch a detached branch"
    assert any("dead" in n for n, _ in rep.failures)

    class Fine(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 4)
        def forward(self, x):
            return self.a(x)

    m2 = Fine()
    l2 = m2(torch.randn(2, 4)).square().mean()
    l2.backward()
    rep2 = check_torch_model(m2, l2)
    print(rep2)
    assert rep2.ok, "arbiter false-positived on a healthy model"
    print("\nself-check PASSED: catches a dead branch, clean on a healthy model")
