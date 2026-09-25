"""Recherche symbolique : quotas de tokens et couverture du budget QSA."""
import torch

from frlm.model_v5 import full_context_indexer, ModelConfigV5Dense
from frlm.prepare_v5 import token_targets


def dense_dimensions(width: int, heads: int, kv_heads: int, head_dim: int, context: int):
    """Validation réelle des dimensions ; aucun tenseur ni appel Transformers."""
    assert -2 <= width <= 4096
    assert -2 <= heads <= 64 and -2 <= kv_heads <= 64
    assert -2 <= head_dim <= 256
    assert -2 <= context <= 4096
    cfg = ModelConfigV5Dense(d_model=width, n_head=heads, n_kv_head=kv_heads,
                             head_dim=head_dim, max_seq_len=context)
    valid = (width > 0 and heads > 0 and kv_heads > 0 and head_dim > 0
             and width == heads * head_dim and heads % kv_heads == 0
             and head_dim % 4 == 0 and width % 8 == 0 and 1 <= context <= 2048)
    try:
        cfg.validate()
    except ValueError:
        assert not valid
    else:
        assert valid


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
