#!/usr/bin/env python3
"""
Batch transcription tool.

Scans an input folder (default: samples/private) for audio files, transcribes each
file using the project's ASR model (chosen via environment variables / config),
and writes outputs to an output folder (default: private_output).

Outputs per file:
  - {basename}.txt  : plain-text transcription
  - {basename}.json : detailed JSON with transcription result and metadata

Also appends a summary line to private_output/summary.csv with:
  filename, audio_seconds, elapsed_seconds, chars, chars_per_second, status

Usage:
  # choose engine/model via env vars, for example:
  ASR_ENGINE=transformers WHISPER_MODEL=openai/whisper-large-v3 python -m src.batch_transcribe
  # or
  ASR_ENGINE=whisperx WHISPERX_MODEL=large-v3 python -m src.batch_transcribe

Notes:
  - For GigaAM via transformers you may need to install additional packages (hydra, omegaconf, torchaudio, etc).
  - The script uses adaptive timeouts implemented in `transcribe_with_timeout`.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Project imports (assumes running from repository root so `src` is importable)
from src.asr.factory import create_asr_model
from src.models.schemas import TranscriptionResponse
from src.services.timeout import transcribe_with_timeout, TranscriptionTimeoutError
from src.utils.audio import load_audio_from_path, get_audio_duration

# Default locations
DEFAULT_INPUT_DIR = Path("samples/private")
DEFAULT_OUTPUT_DIR = Path("private_output")

# Recognized audio globs (recursive)
AUDIO_GLOBS = [
    "**/*.wav",
    "**/*.mp3",
    "**/*.flac",
    "**/*.m4a",
    "**/*.ogg",
    "**/*.mp4",
    "**/*.webm",
    "**/*.aiff",
    "**/*.aac",
]


def find_audio_files(folder: Path) -> List[Path]:
    """Find audio files under folder (recursive) matching known extensions."""
    files = []
    for g in AUDIO_GLOBS:
        files.extend(sorted(folder.glob(g)))
    # Deduplicate and ensure existence
    seen = set()
    unique = []
    for p in files:
        if p.is_file() and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def ensure_output_folder(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_summary_row(csv_path: Path, row: Tuple):
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["filename", "audio_seconds", "elapsed_seconds", "chars", "chars_per_second", "status"])
        w.writerow(row)


def _serialize_result(result) -> dict:
    """Convert result to JSON-serializable dict."""
    if isinstance(result, TranscriptionResponse):
        return result.model_dump(exclude_none=True)
    if isinstance(result, dict):
        return result
    # string or other
    return {"text": str(result)}


def transcribe_file(
    asr_model,
    audio_path: Path,
    output_format: str = "json",
    timeout_enabled: bool = True,
) -> Tuple[Optional[dict], float, str]:
    """
    Transcribe a single file.

    Returns (result_dict_or_None, elapsed_seconds, status)
    status is one of: "ok", "timeout", "error"
    """
    try:
        audio = load_audio_from_path(str(audio_path))
    except Exception as e:
        return ({"error": f"Failed to load audio: {e}"}, 0.0, "load_error")

    audio_seconds = get_audio_duration(audio)

    start = time.perf_counter()
    try:
        if timeout_enabled:
            result, elapsed = transcribe_with_timeout(
                asr_model=asr_model,
                audio=audio,
                audio_duration_sec=audio_seconds,
                task="transcribe",
                language=None,
                word_timestamps=False,
                output=output_format,
            )
            status = "ok"
        else:
            # Direct call (no adaptive timeout)
            # Use the model's transcribe() if returns string/dict or TranscriptionResponse
            asr_model.ensure_model_loaded()
            with asr_model.model_lock:
                raw = asr_model.transcribe(
                    audio=audio,
                    task="transcribe",
                    language=None,
                    word_timestamps=False,
                    output=output_format,
                )
            elapsed = time.perf_counter() - start
            result = raw
            status = "ok"

    except TranscriptionTimeoutError as e:
        elapsed = getattr(e, "elapsed", time.perf_counter() - start)
        result = {"error": f"Transcription timed out after {elapsed:.1f}s", "expected": getattr(e, "expected", None)}
        status = "timeout"
    except Exception as e:
        elapsed = time.perf_counter() - start
        # Include traceback in the JSON for debugging
        tb = traceback.format_exc()
        result = {"error": str(e), "traceback": tb}
        status = "error"

    return (_serialize_result(result), round(elapsed, 3), status)


def main(
    input_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    timeout_enabled: bool = True,
):
    input_dir = input_dir or DEFAULT_INPUT_DIR
    output_dir = output_dir or DEFAULT_OUTPUT_DIR

    print(f"Batch transcription starting")
    print(f"Input folder : {input_dir}")
    print(f"Output folder: {output_dir}")

    if not input_dir.exists():
        print(f"Input folder does not exist: {input_dir}")
        print("Nothing to do.")
        sys.exit(1)

    files = find_audio_files(input_dir)
    if not files:
        print(f"No audio files found in {input_dir}")
        sys.exit(0)

    ensure_output_folder(output_dir)
    summary_csv = output_dir / "summary.csv"

    # Create ASR model
    print("Creating ASR model instance (this uses config from environment)...")
    asr_model = create_asr_model()

    try:
        print("Loading model (this may take a while)...")
        asr_model.load_model()
    except Exception as e:
        print("Failed to load ASR model:")
        print(str(e))
        # Add hint for models like GigaAM that require extra deps
        if "hydra" in str(e).lower() or "omegaconf" in str(e).lower():
            print()
            print("Hint: The selected model's remote code requires additional packages (e.g. hydra/omegaconf/torchaudio).")
            print("You can either:")
            print("  1) Install required packages, for example:")
            print("       pip install hydra-core omegaconf torchaudio sentencepiece")
            print("     (and optionally pyannote.audio for longform support), or")
            print("  2) Install the `gigaam` package locally and use ASR_ENGINE=gigaam:")
            print("       git clone https://github.com/salute-developers/GigaAM.git && cd GigaAM && pip install -e .")
        sys.exit(1)

    # Iterate files
    for idx, f in enumerate(files, start=1):
        print(f"\n[{idx}/{len(files)}] Processing: {f.name}")
        result_dict, elapsed, status = transcribe_file(asr_model, f, output_format="json", timeout_enabled=timeout_enabled)

        # Extract text
        text = result_dict.get("text") if isinstance(result_dict, dict) else str(result_dict)

        # Augment result with metadata
        result_dict["_meta"] = {
            "source_file": str(f),
            "transcription_elapsed_seconds": elapsed,
            "status": status,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Save outputs
        base = f.stem
        txt_path = output_dir / f"{base}.txt"
        json_path = output_dir / f"{base}.json"
        try:
            with txt_path.open("w", encoding="utf-8") as tf:
                tf.write(text or "")
                tf.write("\n")
            with json_path.open("w", encoding="utf-8") as jf:
                json.dump(result_dict, jf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to write outputs for {f}: {e}")
            status = "write_error"

        # Compute some stats
        try:
            audio = load_audio_from_path(str(f))
            audio_sec = get_audio_duration(audio)
        except Exception:
            audio_sec = None

        chars = len(text) if text else 0
        cps = round(chars / audio_sec, 4) if audio_sec and audio_sec > 0 else None

        append_summary_row(summary_csv, (str(f), audio_sec, elapsed, chars, cps, status))

        print(f"-> Saved: {json_path} (elapsed={elapsed}s, chars={chars}, cps={cps}, status={status})")

    # Final cleanup
    try:
        asr_model.release_model()
    except Exception:
        pass

    print("\nBatch transcription completed.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Batch transcription of samples/private -> private_output")
    p.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT_DIR, help="Input directory (default: samples/private)")
    p.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory (default: private_output)")
    p.add_argument("--no-timeout", dest="timeout", action="store_false", help="Disable adaptive timeout (use direct model.transcribe())")
    args = p.parse_args()

    main(input_dir=args.input, output_dir=args.output, timeout_enabled=args.timeout)
