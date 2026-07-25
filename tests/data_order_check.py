"""Data-ordering check — catches domain-grouped shard order.

An early loader read shards grouped by domain. The loss curve sawtoothed at every domain
boundary as the model serially forgot each domain in turn. Interleave, then assert no
window is dominated by one domain.
"""
from collections import Counter


def assert_domain_mixed(domain_sequence, window=64, max_share=0.60):
    n = len(domain_sequence)
    assert n >= window, f"sequence shorter than window ({n} < {window})"
    worst, worst_at = 0.0, -1
    for i in range(n - window + 1):
        dom, cnt = Counter(domain_sequence[i:i+window]).most_common(1)[0]
        share = cnt / window
        if share > worst: worst, worst_at = share, i
    assert worst <= max_share, (
        f"window at index {worst_at} is {worst:.0%} one domain (max {max_share:.0%}) "
        f"-- shards are domain-grouped, expect sawtooth forgetting")
    return worst


if __name__ == "__main__":
    import itertools
    grouped = ["bn"]*100 + ["en"]*100
    try:
        assert_domain_mixed(grouped); raise SystemExit("FAILED: missed grouped order")
    except AssertionError as e:
        print("caught domain-grouped order:", str(e)[:72])
    mixed = list(itertools.islice(itertools.cycle(["bn","en"]), 200))  # balanced 50/50
    print(f"interleaved worst-window share: {assert_domain_mixed(mixed):.0%}")
    print("data-ordering check: PASS")
