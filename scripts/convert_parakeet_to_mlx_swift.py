"""Конвертация Parakeet-TDT-v3 (mlx-community) для Swift-порта oaitt.

Готовит data/parakeet_tdt_v3/:
- weights.safetensors  - веса модели в bfloat16 (ключи совпадают с mlx-community)
- filterbanks.safetensors - mel-фильтрбанк librosa (slaney), который parakeet_mlx
  считает в PreprocessArgs.__post_init__; в Swift считаем один раз в Python, чтобы
  гарантированно совпасть с эталоном
- vocab.txt - словарь 8192 токенов из config.json (joint.vocabulary), по строке на токен

Запуск: venv/bin/python scripts/convert_parakeet_to_mlx_swift.py
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "parakeet_tdt_v3"


def main() -> int:
    import librosa
    import mlx.core as mx
    import numpy as np
    from huggingface_hub import snapshot_download

    OUT.mkdir(parents=True, exist_ok=True)

    snapshot = Path(snapshot_download("mlx-community/parakeet-tdt-0.6b-v3"))
    config = json.load(open(snapshot / "config.json"))

    # Веса в bfloat16 - тот же dtype, что from_pretrained в parakeet_mlx.
    weights = mx.load(str(snapshot / "model.safetensors"))
    casted = {k: v.astype(mx.bfloat16) for k, v in weights.items()}
    mx.save_safetensors(str(OUT / "weights.safetensors"), casted)
    print(f"weights: {len(casted)} tensors -> {OUT / 'weights.safetensors'}")

    # Mel-фильтрбанк ровно тот, что строит parakeet_mlx (librosa, slaney).
    pp = config["preprocessor"]
    assert pp["sample_rate"] == 16000 and pp["features"] == 128
    banks = librosa.filters.mel(
        sr=pp["sample_rate"],
        n_fft=pp["n_fft"],
        n_mels=pp["features"],
        fmin=0,
        fmax=pp["sample_rate"] / 2,
        norm="slaney",
    )
    mx.save_safetensors(
        str(OUT / "filterbanks.safetensors"), {"filterbanks": mx.array(banks)}
    )
    print(f"filterbanks: {banks.shape} -> {OUT / 'filterbanks.safetensors'}")

    vocab = config["joint"]["vocabulary"]
    (OUT / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    print(f"vocab: {len(vocab)} tokens -> {OUT / 'vocab.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
