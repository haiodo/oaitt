"""WER-оценка ASR-моделей на Golos (bond005/sberdevices_golos_10h_crowd, test-сплит).

Протокол тот же, что в docs/benchmarks.md: первые 395 записей test-сплита, нормализация
«числа прописью» (num2words), сравнение в лоб и на подмножестве без латиницы.

Модели и их зависимости (ставить в отдельный venv, не в основной):
    gigaam    - gigaam_mlx из vendor/ (mlx, sentencepiece, librosa)
    qwen3     - mlx-audio (mlx-community/Qwen3-ASR-1.7B-6bit)
    parakeet  - parakeet-mlx (mlx-community/parakeet-tdt-0.6b-v3)
Общие: datasets, jiwer, num2words, soundfile

Запуск:
    python scripts/eval_golos_wer.py gigaam
    python scripts/eval_golos_wer.py qwen3 --model-id mlx-community/Qwen3-ASR-1.7B-8bit
    python scripts/eval_golos_wer.py parakeet
    python scripts/eval_golos_wer.py compare --hyps-dir /tmp
"""

import argparse
import io
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR_GIGAAM = os.path.join(REPO, "vendor", "gigaam-mlx")


def load_transcriber(kind: str, model_id: str | None):
    if kind == "gigaam":
        sys.path.insert(0, VENDOR_GIGAAM)
        from gigaam_mlx import load_model as gm_load, transcribe as gm_transcribe

        repo = model_id or os.path.join(REPO, "data", "gigaam_mlx", "rnnt")
        model, tok = gm_load("rnnt", repo_id=repo)
        return lambda path: gm_transcribe(model, tok, path)

    if kind == "qwen3":
        from mlx_audio.stt.generate import generate_transcription, load_model

        model = load_model(model_id or "mlx-community/Qwen3-ASR-1.7B-6bit")
        return lambda path: generate_transcription(model=model, audio=path).text

    if kind == "parakeet":
        from parakeet_mlx import from_pretrained

        model = from_pretrained(model_id or "mlx-community/parakeet-tdt-0.6b-v3")
        return lambda path: model.transcribe(path).text

    raise ValueError(f"unknown model kind: {kind}")


def to_words_num(s: str) -> str:
    import num2words

    def repl(m):
        try:
            return num2words.num2words(int(m.group(0)), lang="ru")
        except Exception:
            return m.group(0)

    return re.sub(r"\d+", repl, s)


def normalize(s: str) -> str:
    s = to_words_num(s.lower().replace("ё", "е"))
    s = re.sub(r"[^а-яa-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def has_latin(s: str) -> bool:
    return bool(re.search(r"[a-z]", s))


def run_eval(kind: str, model_id: str | None, n: int, out_path: str):
    import soundfile as sf
    from datasets import Audio, load_dataset

    transcribe = load_transcriber(kind, model_id)

    ds = load_dataset("bond005/sberdevices_golos_10h_crowd", split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.select(range(min(n, len(ds))))

    refs, hyps = [], []
    for i, rec in enumerate(ds):
        if not rec.get("transcription"):
            continue
        a = rec["audio"]
        if not a or not a.get("bytes"):
            continue
        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sf.write("/tmp/_golos_cur.wav", wav, 16000)
        refs.append(rec["transcription"])
        hyps.append(transcribe("/tmp/_golos_cur.wav"))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(ds)}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"refs": refs, "hyps": hyps}, f, ensure_ascii=False)
    print(f"saved {len(refs)} pairs -> {out_path}", flush=True)


def compare(hyps_dir: str):
    import jiwer

    models = ["gigaam", "qwen3", "qwen3-8bit", "parakeet"]
    data = {}
    for m in models:
        path = os.path.join(hyps_dir, f"golos_hyps_{m}.json")
        if os.path.exists(path):
            data[m] = json.load(open(path))

    ref0 = data["gigaam"]["refs"]
    base_idx = [i for i, r in enumerate(ref0) if not has_latin(normalize(r))]
    print(f"records total={len(ref0)}, ref-latin-free={len(base_idx)}")

    for m, d in data.items():
        refs = [normalize(r) for r in d["refs"]]
        hyps = [normalize(h) for h in d["hyps"]]
        common = jiwer.wer([refs[i] for i in base_idx], [hyps[i] for i in base_idx])
        pairs = [(r, h) for r, h in zip(refs, hyps) if not has_latin(r) and not has_latin(h)]
        per_model = jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])
        print(
            f"{m:12s} common-subset WER={common * 100:.2f}%  "
            f"(per-model-subset={per_model * 100:.2f}%, n={len(pairs)})"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=["gigaam", "qwen3", "parakeet", "compare"])
    p.add_argument("--model-id", default=None, help="HF repo id или локальный путь")
    p.add_argument("--n", type=int, default=395)
    p.add_argument("--tag", default=None, help="имя файла результата (по умолчанию - kind)")
    p.add_argument("--hyps-dir", default="/tmp", help="куда класть / откуда читать гипотезы")
    args = p.parse_args()

    if args.kind == "compare":
        compare(args.hyps_dir)
        return

    tag = args.tag or args.kind
    run_eval(args.kind, args.model_id, args.n, os.path.join(args.hyps_dir, f"golos_hyps_{tag}.json"))


if __name__ == "__main__":
    main()
