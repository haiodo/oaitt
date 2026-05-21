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
import logging
import os
import shutil
import sys

# Project root (when run as a script)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
DEFAULT_GIGAAM_CACHE = os.path.join(ROOT, "data", "gigaam")
DEFAULT_MLX_OUTPUT_ROOT = os.path.join(ROOT, "data", "gigaam_mlx")

# Make both submodules importable when running as a script
sys.path.insert(0, os.path.join(ROOT, "vendor", "gigaam"))
sys.path.insert(0, os.path.join(ROOT, "vendor", "gigaam-mlx"))

logger = logging.getLogger(__name__)


def convert_one(
    model_type: str,
    gigaam_cache: str = DEFAULT_GIGAAM_CACHE,
    output_root: str = DEFAULT_MLX_OUTPUT_ROOT,
) -> str:
    """
    Конвертирует одну модель (ctc или rnnt) из PyTorch GigaAM в MLX safetensors.

    Args:
        model_type: "ctc" или "rnnt"
        gigaam_cache: каталог с .ckpt файлами (data/gigaam/)
        output_root: корень выходных каталогов (data/gigaam_mlx/)

    Returns:
        Путь к директории с MLX весами.
    """
    if model_type not in ("ctc", "rnnt"):
        raise ValueError(f"model_type must be ctc|rnnt, got {model_type!r}")

    gigaam_name = f"v3_e2e_{model_type}"
    out_dir = os.path.join(output_root, model_type)
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

    logger.info(f"Loading PyTorch GigaAM {gigaam_name} from {gigaam_cache}...")
    if not os.path.isdir(gigaam_cache):
        raise FileNotFoundError(
            f"Local cache {gigaam_cache} not found. "
            "Run ./run_gigaam_asr.sh once to download weights."
        )

    pt_model = gigaam.load_model(gigaam_name, download_root=gigaam_cache)
    pt_state = {k: v for k, v in pt_model.named_parameters()}
    for k, v in pt_model.named_buffers():
        pt_state[k] = v

    logger.info("Converting encoder weights...")
    weights = convert_encoder(pt_state)

    logger.info(f"Converting {model_type} head weights...")
    if model_type == "ctc":
        weights.update(convert_ctc_head(pt_state))
    else:
        weights.update(convert_rnnt_head(pt_state))

    mlx_weights = {k: mx.array(v) for k, v in weights.items()}

    weights_path = os.path.join(out_dir, "weights.safetensors")
    logger.info(f"Saving weights -> {weights_path}")
    mx.save_safetensors(weights_path, mlx_weights)

    tokenizer_src = os.path.join(gigaam_cache, f"{gigaam_name}_tokenizer.model")
    tokenizer_dst = os.path.join(out_dir, "tokenizer.model")
    if os.path.exists(tokenizer_src):
        shutil.copy2(tokenizer_src, tokenizer_dst)
        logger.info(f"Copied tokenizer -> {tokenizer_dst}")
    else:
        logger.warning(f"tokenizer not found at {tokenizer_src}")

    total = sum(v.size for v in mlx_weights.values())
    logger.info(f"Done. {len(mlx_weights)} tensors, {total:,} parameters -> {out_dir}")

    # Free PyTorch model
    del pt_model
    del pt_state
    del weights
    del mlx_weights

    return out_dir


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
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
        logger.info(f"\n=== Converting {t} ===")
        out = convert_one(t)
        logger.info(f"OK: {out}")

    logger.info("\nTo use locally converted weights, set:")
    for t in targets:
        logger.info(f"  GIGAAM_MLX_REPO_ID={os.path.join(DEFAULT_MLX_OUTPUT_ROOT, t)} ./run_gigaam_mlx_asr.sh")


if __name__ == "__main__":
    main()
