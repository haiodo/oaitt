#!/usr/bin/env python3
"""
Конвертирует локальные веса GigaAM (data/gigaam/) в MLX safetensors формат.

Использует уже скачанные .ckpt файлы из data/gigaam/ (загружены движком gigaam ASR)
и upstream convert.py из vendor/gigaam-mlx. Результат: data/gigaam_mlx/{ctc,rnnt}/.

После запуска можно использовать локальный путь без скачивания с HuggingFace:
    GIGAAM_MLX_REPO_ID=./data/gigaam_mlx/ctc ./run_gigaam_mlx_asr.sh

Usage:
    python scripts/convert_gigaam_to_mlx.py                  # convert ctc (default)
    python scripts/convert_gigaam_to_mlx.py --model rnnt     # convert rnnt
    python scripts/convert_gigaam_to_mlx.py --model both     # convert both

Copyright (c) 2026 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import argparse
import os
import shutil
import sys

# Project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GIGAAM_CACHE = os.path.join(ROOT, "data", "gigaam")
MLX_OUTPUT_ROOT = os.path.join(ROOT, "data", "gigaam_mlx")

# Make both submodules importable
sys.path.insert(0, os.path.join(ROOT, "vendor", "gigaam"))
sys.path.insert(0, os.path.join(ROOT, "vendor", "gigaam-mlx"))


def convert_one(model_type: str) -> str:
    """
    Конвертирует одну модель (ctc или rnnt).

    Args:
        model_type: "ctc" или "rnnt"

    Returns:
        Путь к директории с MLX весами.
    """
    if model_type not in ("ctc", "rnnt"):
        raise ValueError(f"model_type must be ctc|rnnt, got {model_type!r}")

    gigaam_name = f"v3_e2e_{model_type}"
    out_dir = os.path.join(MLX_OUTPUT_ROOT, model_type)
    os.makedirs(out_dir, exist_ok=True)

    # SSL certs (matches upstream behavior)
    try:
        import ssl
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
        ssl._create_default_https_context = lambda: ssl.create_default_context(
            cafile=certifi.where()
        )
    except ImportError:
        pass

    import mlx.core as mx
    import gigaam
    from gigaam_mlx.convert import (
        convert_encoder,
        convert_ctc_head,
        convert_rnnt_head,
    )

    print(f"Loading PyTorch GigaAM {gigaam_name} from {GIGAAM_CACHE}...")
    if not os.path.isdir(GIGAAM_CACHE):
        raise FileNotFoundError(
            f"Local cache {GIGAAM_CACHE} not found. "
            "Run ./run_gigaam_asr.sh once to download weights."
        )

    pt_model = gigaam.load_model(gigaam_name, download_root=GIGAAM_CACHE)
    pt_state = {k: v for k, v in pt_model.named_parameters()}
    for k, v in pt_model.named_buffers():
        pt_state[k] = v

    print("Converting encoder weights...")
    weights = convert_encoder(pt_state)

    print(f"Converting {model_type} head weights...")
    if model_type == "ctc":
        weights.update(convert_ctc_head(pt_state))
    else:
        weights.update(convert_rnnt_head(pt_state))

    mlx_weights = {k: mx.array(v) for k, v in weights.items()}

    weights_path = os.path.join(out_dir, "weights.safetensors")
    print(f"Saving weights -> {weights_path}")
    mx.save_safetensors(weights_path, mlx_weights)

    tokenizer_src = os.path.join(GIGAAM_CACHE, f"{gigaam_name}_tokenizer.model")
    tokenizer_dst = os.path.join(out_dir, "tokenizer.model")
    if os.path.exists(tokenizer_src):
        shutil.copy2(tokenizer_src, tokenizer_dst)
        print(f"Copied tokenizer -> {tokenizer_dst}")
    else:
        print(f"WARNING: tokenizer not found at {tokenizer_src}")

    total = sum(v.size for v in mlx_weights.values())
    print(f"Done. {len(mlx_weights)} tensors, {total:,} parameters -> {out_dir}")

    # Free PyTorch model
    del pt_model
    del pt_state
    del weights
    del mlx_weights

    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert local GigaAM PyTorch weights to MLX format"
    )
    parser.add_argument(
        "--model",
        choices=["ctc", "rnnt", "both"],
        default="ctc",
        help="Which model variant to convert (default: ctc)",
    )
    args = parser.parse_args()

    targets = ["ctc", "rnnt"] if args.model == "both" else [args.model]

    for t in targets:
        print(f"\n=== Converting {t} ===")
        out = convert_one(t)
        print(f"OK: {out}")

    print("\nTo use locally converted weights, set:")
    for t in targets:
        print(f"  GIGAAM_MLX_REPO_ID={os.path.join(MLX_OUTPUT_ROOT, t)} ./run_gigaam_mlx_asr.sh")


if __name__ == "__main__":
    main()
