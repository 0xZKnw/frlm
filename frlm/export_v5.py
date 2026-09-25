"""Export HF/GGUF de la v5 Qwen3.5 dense et des anciens modèles.

Les seules adaptations sont le BPE français reconnu par sa structure, le nom
canonique des couches QSA et la table PLE non fragmentée de Transformers 5.17.
La variante Qwen4-exp dense historique s'exporte avec --hf-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

LLAMA_REVISION = "a72e04abe0fe9b36e203033ac71bd5f379c35bc5"


def export_hf(checkpoint: Path, tokenizer: Path, output: Path):
    import torch
    from transformers import PreTrainedTokenizerFast
    from frlm import config_from_dict, model_from_cfg
    from frlm import data as D

    if output.exists():
        raise FileExistsError(f"export déjà présent : {output}")
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = config_from_dict(ck["model_cfg"])
    if cfg.to_dict().get("arch") not in ("v5-qwen4exp", "v5-dense", "v5-qwen35"):
        raise ValueError("checkpoint autre que v5")
    from frlm.prepare_v5 import sha256
    if ck.get("tokenizer_sha256") != sha256(tokenizer):
        raise ValueError("empreinte du tokenizer différente du checkpoint")
    model = model_from_cfg(cfg)
    model.load_state_dict(ck["model"])
    fast = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer), eos_token=D.EOT,
                                   bos_token=D.EOT, pad_token=D.EOT,
                                   additional_special_tokens=D.SPECIALS[1:])
    if len(fast) != cfg.vocab_size:
        raise ValueError("tokenizer incompatible avec la configuration")
    model.hf.generation_config.eos_token_id = [fast.convert_tokens_to_ids(s) for s in (D.EOT, D.IM_END)]
    fast.chat_template = "{% for m in messages %}{% set c = m['content']|trim %}{% if c %}{{ '<|im_start|>' + m['role'] + '\n' + c + '<|im_end|>\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hf-v5-", dir=output.parent) as tmp:
        staging = Path(tmp) / "hf"
        model.hf.save_pretrained(staging)
        fast.save_pretrained(staging)
        normalize_config(staging)
        staging.rename(output)


def normalize_config(folder: Path):
    path = folder / "config.json"
    cfg = json.loads(path.read_text())
    if cfg["model_type"] in ("frlm_qwen4exp_dense", "qwen3_5_text"):
        return  # Export HF natif : ne pas appliquer les adaptations GGUF historiques.
    cfg["layer_types"] = ["full_attention" if t == "qwen_sparse_attention" else t
                          for t in cfg["layer_types"]]
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def convert(folder: Path, output: Path, llama_cpp: Path, outtype="f32"):
    from frlm.prepare_v5 import QWEN2_PATTERN

    if output.exists():
        raise FileExistsError(f"GGUF déjà présent : {output}")
    cfg = json.loads((folder / "config.json").read_text())
    if cfg["model_type"] == "frlm_qwen4exp_dense":
        raise ValueError("GGUF Qwen4-exp dense non pris en charge ; utiliser --hf-only")
    revision = subprocess.check_output(["git", "-C", str(llama_cpp), "rev-parse", "HEAD"], text=True).strip()
    if revision != LLAMA_REVISION:
        raise ValueError(f"llama.cpp attendu : {LLAMA_REVISION}, reçu : {revision}")
    backend = json.loads((folder / "tokenizer.json").read_text())
    expected = {"type": "Sequence", "pretokenizers": [
        {"type": "Split", "pattern": {"Regex": QWEN2_PATTERN}, "behavior": "Isolated", "invert": False},
        {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False},
    ]}
    if backend["pre_tokenizer"] != expected or backend.get("normalizer") is not None:
        raise ValueError("pré-tokeniseur non vérifié pour l'export qwen2 GGUF")
    previous_path, previous_argv = sys.path[:], sys.argv[:]
    sys.path[:0] = [str(llama_cpp), str(llama_cpp / "gguf-py")]
    import gguf
    if cfg["model_type"] == "qwen3_5_text":
        from conversion.qwen import Qwen3_5TextModel as converter
    else:
        from conversion.qwen4exp import Qwen4ExpTextModel as converter

    # Le vocabulaire français neuf conserve exactement la pré-tokenisation qwen2.
    previous_vocab = converter.get_vocab_base_pre
    converter.get_vocab_base_pre = lambda self, tokenizer: "qwen2"
    original = converter.modify_tensors
    if cfg["model_type"] != "qwen3_5_text":
        def tensors(self, tensor, name, bid):
            if name.endswith(".ple_embedding.ngram_embedding.weight"):
                self._ple_row_dim = int(tensor.shape[-1])
                return [(gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD] + ".weight", tensor)]
            return original(self, tensor, name, bid)
        converter.modify_tensors = tensors
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".gguf-v5-", dir=output.parent) as tmp:
            staging = Path(tmp) / output.name
            sys.argv = ["convert_hf_to_gguf.py", str(folder), "--outfile", str(staging), "--outtype", outtype]
            if cfg["model_type"] == "qwen3_5_text":
                sys.argv.append("--no-mtp")
            runpy.run_path(str(llama_cpp / "convert_hf_to_gguf.py"), run_name="__main__")
            staging.rename(output)
    finally:
        converter.modify_tensors = original
        converter.get_vocab_base_pre = previous_vocab
        sys.path[:], sys.argv = previous_path, previous_argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tokenizer", type=Path, default=Path("data-v5/tokenizer.json"))
    p.add_argument("--hf-dir", type=Path, required=True)
    p.add_argument("--hf-only", action="store_true", help="export HF uniquement")
    p.add_argument("--llama-cpp", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--outtype", choices=["f32", "f16", "bf16", "q8_0"], default="f16")
    args = p.parse_args()
    if args.hf_only:
        if not args.checkpoint:
            p.error("--hf-only exige --checkpoint")
        export_hf(args.checkpoint, args.tokenizer, args.hf_dir)
        return
    if args.llama_cpp is None or args.output is None:
        p.error("GGUF exige --llama-cpp et --output")
    if args.checkpoint:
        export_hf(args.checkpoint, args.tokenizer, args.hf_dir)
    convert(args.hf_dir, args.output, args.llama_cpp, args.outtype)


if __name__ == "__main__":
    main()
