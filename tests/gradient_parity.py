"""Gradient-parity check — caught defect D4b within minutes.

A fused kernel and a hand-rolled reference implementation of the same maths must agree on
BOTH forward value and gradient. D4b: a correct backward was written for a fused softmax,
its own unit tests passed, and training still used the old no-backward path -- because the
public forward function called the wrong entry point. Forward parity alone would not have
caught it; gradient parity did.
"""
import torch


def assert_grad_parity(fused_fn, reference_fn, make_input, atol=1e-5, name="op"):
    x1 = make_input(); x1.requires_grad_(True)
    x2 = x1.detach().clone().requires_grad_(True)
    y1, y2 = fused_fn(x1), reference_fn(x2)
    assert torch.allclose(y1, y2, atol=atol), f"{name}: FORWARD differs (max {(y1-y2).abs().max():.2e})"
    # NB: a plain .sum() is a vacuous probe for anything that sums to a constant
    # (softmax!) -- its true gradient is zero, so a zero-returning backward would
    # pass. Use a non-degenerate reduction.
    w = torch.linspace(1.0, 2.0, y1.shape[-1], dtype=y1.dtype)
    (y1 * w).sum().backward(); (y2 * w).sum().backward()
    assert x1.grad is not None, f"{name}: fused path produced NO gradient"
    d = (x1.grad - x2.grad).abs().max()
    assert d < atol, f"{name}: GRADIENT differs (max {d:.2e}) -- fused backward is wrong or unused"
    return True


if __name__ == "__main__":
    ref = lambda t: (t - t.max(-1, keepdim=True).values).exp() / \
                    (t - t.max(-1, keepdim=True).values).exp().sum(-1, keepdim=True)
    assert_grad_parity(lambda t: torch.softmax(t, -1), ref,
                       lambda: torch.randn(3, 7, dtype=torch.float64), name="softmax")
    print("gradient parity (softmax fused vs reference): PASS")

    class NoBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, t): return torch.softmax(t, -1)
        @staticmethod
        def backward(ctx, g): return torch.zeros_like(g)   # the D4b failure mode
    try:
        assert_grad_parity(NoBackward.apply, ref,
                           lambda: torch.randn(3, 7, dtype=torch.float64), name="broken")
        raise SystemExit("FAILED: parity check missed a wrong backward")
    except AssertionError as e:
        print("caught the D4b failure mode:", str(e)[:70])
