"""Recherche symbolique : quotas de tokens et couverture du budget QSA."""
import torch

from frlm.model_v5 import full_context_indexer, ModelConfigV5Dense, ModelConfigV5Qwen35
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


def qwen35_head_dimensions(width: int, heads: int, kv_heads: int, head_dim: int):
    """Le preset texte natif accepte exactement les têtes GQA/MRoPE cohérentes."""
    assert -1 <= width <= 1024
    assert -1 <= heads <= 32 and -1 <= kv_heads <= 16
    assert -1 <= head_dim <= 128
    cfg = ModelConfigV5Qwen35(d_model=width, n_head=heads, n_kv_head=kv_heads,
                              head_dim=head_dim)
    valid = (width > 0 and heads > 0 and kv_heads > 0 and head_dim > 0
             and width == heads * head_dim and heads % kv_heads == 0
             and width % 8 == 0 and head_dim % 8 == 0)
    try:
        cfg.validate()
    except ValueError:
        assert not valid
    else:
        assert valid


def qwen35_layer_dimensions(layers: int, context: int, value_heads: int, key_heads: int):
    assert -1 <= layers <= 32 and -1 <= context <= 2049
    assert -1 <= value_heads <= 16 and -1 <= key_heads <= 16
    cfg = ModelConfigV5Qwen35(n_layer=layers, max_seq_len=context,
                              linear_heads=value_heads, linear_key_heads=key_heads)
    valid = (layers >= 4 and layers % 4 == 0 and 1 <= context <= 2048
             and value_heads > 0 and key_heads > 0 and value_heads % key_heads == 0)
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
