"""Position-effect probe — the test that caught the rotary-axis bug in §3.2.

Rotary embeddings applied on the head axis instead of the sequence axis give EVERY position
an identical rotation: the model has no positional signal at all, and no amount of further
training fixes it. Code review missed this more than once. A test of the *effect* caught it.

Principle: destroy input token order. A model with working positional information must get
measurably worse. If the loss barely moves, positional encoding is not doing anything.
"""
import torch


def probe_position_sensitivity(loss_fn, tokens, min_ratio=1.10, seed=0):
    """loss_fn(tokens) -> scalar loss. Returns (ordered, shuffled, ratio, ok)."""
    g = torch.Generator().manual_seed(seed)
    ordered = float(loss_fn(tokens))
    perm = torch.randperm(tokens.shape[-1], generator=g)
    shuffled = float(loss_fn(tokens[..., perm]))
    ratio = shuffled / max(ordered, 1e-9)
    return ordered, shuffled, ratio, ratio >= min_ratio


if __name__ == "__main__":
    # positive control: a model that USES order degrades when order is destroyed
    def order_sensitive(t):
        return float((t[..., 1:] - t[..., :-1]).float().abs().mean())
    # negative control: a bag-of-tokens model is order-blind (the bug's signature)
    def order_blind(t):
        return float(t.float().mean())

    toks = torch.arange(64).unsqueeze(0)
    for name, fn, expect_ok in (("order-sensitive", order_sensitive, True),
                                ("order-blind (bug signature)", order_blind, False)):
        o, s, r, ok = probe_position_sensitivity(fn, toks)
        print(f"  {name:30s} ordered={o:.4f} shuffled={s:.4f} ratio={r:.3f} sensitive={ok}")
        assert ok == expect_ok, f"probe gave the wrong verdict for {name}"
    print("position-effect probe: PASS (flags an order-blind model, passes an order-sensitive one)")
