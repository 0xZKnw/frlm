"""Interpolation linéaire de deux checkpoints compatibles, sans optimiseur.

Utile pour chercher localement un compromis entre une politique SFT conservée et
un petit déplacement de bootstrap, sans nouvelle allocation GPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def blend(base_path: Path, tuned_path: Path, output: Path, alpha: float) -> dict:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha doit être compris entre 0 et 1")
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    tuned = torch.load(tuned_path, map_location="cpu", weights_only=False)
    if base["model_cfg"] != tuned["model_cfg"]:
        raise ValueError("configurations modèle incompatibles")
    if set(base["model"]) != set(tuned["model"]):
        raise ValueError("state_dict incompatibles")
    state = {}
    for name, tuned_tensor in tuned["model"].items():
        base_tensor = base["model"][name]
        if base_tensor.shape != tuned_tensor.shape or base_tensor.dtype != tuned_tensor.dtype:
            raise ValueError(f"tenseur incompatible : {name}")
        if torch.is_floating_point(tuned_tensor):
            state[name] = torch.lerp(base_tensor.float(), tuned_tensor.float(), alpha).to(
                tuned_tensor.dtype
            )
        else:
            if not torch.equal(base_tensor, tuned_tensor):
                raise ValueError(f"buffer non flottant divergent : {name}")
            state[name] = tuned_tensor.clone()
    payload = {
        "model": state, "model_cfg": tuned["model_cfg"], "stage": "sft",
        "step": int(tuned.get("step", 0)),
        "tokens_seen": int(tuned.get("tokens_seen", 0)),
        "val_loss": float("nan"), "best_val": float("nan"),
        "blend": {"base": str(base_path), "tuned": str(tuned_path), "alpha": alpha},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(output)
    return payload["blend"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    args = parser.parse_args()
    info = blend(Path(args.base), Path(args.tuned), Path(args.output), args.alpha)
    print(f"[ok] Checkpoint interpolé : {args.output} · alpha={info['alpha']:.3f}")


if __name__ == "__main__":
    main()
