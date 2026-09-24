"""Recherche symbolique : quotas de tokens et couverture du budget QSA."""
import torch

from frlm.model_v5 import full_context_indexer
from frlm.prepare_v5 import token_targets


def qsa_full_budget(keys: int, visible: int, budget: int, block: int):
    """Branche réelle du fast path + cardinalité de la sélection QSA amont.

    Les tenseurs/cache et les valeurs numériques sont couverts séparément par tests.
    """
    assert 1 <= keys <= 2048
    assert 0 <= visible <= keys
    assert 1 <= budget <= 2048
    assert 1 <= block <= 32

    class Mask:
        dtype = torch.bool
        shape = (keys,)

    class Indexer:
        token_budget = budget

        def forward(self, hidden, positions, mask, cache):
            return None  # sentinelle du repli amont, sans E/S

    mask = Mask()
    selected = full_context_indexer(Indexer(), None, None, mask, None)
    if selected is mask:
        # L'amont conserve tous les blocs sélectionnés et le reliquat.
        kept = min(budget // block, visible // block) * block + visible % block
        assert kept == visible
    else:
        assert keys > budget


def quotas(total: int, a: int, b: int, c: int):
    assert 0 < total <= 1_000_000_000_000
    assert 0 < a <= 100 and 0 < b <= 100 and 0 < c <= 100
    counts = token_targets(total, [a, b, c])
    assert sum(counts) == total
    for n, w in zip(counts, [a, b, c]):
        assert n >= 0
        assert abs(n * (a + b + c) - total * w) < a + b + c
