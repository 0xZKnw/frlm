"""Recherche symbolique sur l'allocation réelle des tokens ; aucune E/S."""
from frlm.prepare_v5 import token_targets


def quotas(total: int, a: int, b: int, c: int):
    assert 0 < total <= 1_000_000_000_000
    assert 0 < a <= 100 and 0 < b <= 100 and 0 < c <= 100
    counts = token_targets(total, [a, b, c])
    assert sum(counts) == total
    for n, w in zip(counts, [a, b, c]):
        assert n >= 0
        assert abs(n * (a + b + c) - total * w) < a + b + c
