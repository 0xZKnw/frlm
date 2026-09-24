"""Export HF/GGUF de la v5 ; réutilise le convertisseur officiel llama.cpp.

Les seules adaptations sont le BPE français reconnu par sa structure, le nom
canonique des couches QSA et la table PLE non fragmentée de Transformers 5.17.
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
    if cfg.to_dict().get("arch") != "v5-qwen4exp":
        raise ValueError("checkpoint autre que Qwen4-exp v5")
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
    cfg["layer_types"] = ["full_attention" if t == "qwen_sparse_attention" else t
                          for t in cfg["layer_types"]]
    path.write_text(json.dumps(cfg, indent=2) + "\n")


def convert(folder: Path, output: Path, llama_cpp: Path, outtype="f32"):
    from frlm.prepare_v5 import QWEN2_PATTERN

    if output.exists():
        raise FileExistsError(f"GGUF déjà présent : {output}")
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
    from conversion.qwen4exp import Qwen4ExpTextModel

    # La détection amont dépend d'un hash d'IDs : impossible pour un vocabulaire neuf.
    previous_vocab = Qwen4ExpTextModel.get_vocab_base_pre
    Qwen4ExpTextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen2"
    original = Qwen4ExpTextModel.modify_tensors

    def tensors(self, tensor, name, bid):
        if name.endswith(".ple_embedding.ngram_embedding.weight"):
            self._ple_row_dim = int(tensor.shape[-1])
            return [(gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD] + ".weight", tensor)]
        return original(self, tensor, name, bid)

    Qwen4ExpTextModel.modify_tensors = tensors
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".gguf-v5-", dir=output.parent) as tmp:
            staging = Path(tmp) / output.name
            sys.argv = ["convert_hf_to_gguf.py", str(folder), "--outfile", str(staging), "--outtype", outtype]
            runpy.run_path(str(llama_cpp / "convert_hf_to_gguf.py"), run_name="__main__")
            staging.rename(output)
    finally:
        Qwen4ExpTextModel.modify_tensors = original
        Qwen4ExpTextModel.get_vocab_base_pre = previous_vocab
        sys.path[:], sys.argv = previous_path, previous_argv


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tokenizer", type=Path, default=Path("data-v5/tokenizer.json"))
    p.add_argument("--hf-dir", type=Path, required=True)
    p.add_argument("--llama-cpp", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--outtype", choices=["f32", "f16", "bf16", "q8_0"], default="f16")
    args = p.parse_args()
    if args.checkpoint:
        export_hf(args.checkpoint, args.tokenizer, args.hf_dir)
    convert(args.hf_dir, args.output, args.llama_cpp, args.outtype)


if __name__ == "__main__":
    main()
