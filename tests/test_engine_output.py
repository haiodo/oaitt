#!/usr/bin/env python3
"""
OAITT — Open AI Transformer Transcriber.

Проверка валидности вывода ASR движков.

Ловит поломки контракта с апстримом: например, gigaam >= 0.2 стал возвращать
из decode() кортеж (text, token_ids, token_frames) вместо строки, и сырой
кортеж утекал в текст транскрипции - ответ приходил с кодом 200, а внутри
был мусор вида "('привет', [43, 44, ...])".

Для каждого движка поднимает сервер, транскрибирует эталонное аудио и
проверяет, что текст действительно похож на русскую речь.

Copyright (c) 2026 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.

Usage:
    python -m tests.test_engine_output               # все движки
    python -m tests.test_engine_output --engine "GigaAM Native (CTC)"
"""

import argparse
import re
import sys
from pathlib import Path

from tests.test_benchmark import (
    BENCHMARK_SCRIPTS,
    PROJECT_ROOT,
    SERVER_PORT,
    get_test_audio,
    is_port_in_use,
    kill_process_on_port,
    reset_session,
    start_server,
    stop_server,
    transcribe_audio,
    wait_for_port_free,
    wait_for_server,
)

# Эталонное аудио начинается со слов "проверяем транскрипцию через гигачат".
# Ищем короткие устойчивые куски: модели без пунктуации путают безударные
# гласные ("транскрепцию"), поэтому целые слова брать нельзя.
EXPECTED_SUBSTRINGS = ["провер", "транскр"]

# Минимальная длина текста: аудио 137s, даже самая скупая модель даёт сотни
# символов. Пустой или обрезанный ответ так не проскочит.
MIN_TEXT_LENGTH = 200

# Признаки утечки Python-репрезентации в текст: кортежи, списки токенов,
# объекты dataclass. Именно так выглядела поломка контракта gigaam.
LEAK_PATTERNS = [
    (re.compile(r"^\s*\("), "text starts with a tuple repr"),
    (re.compile(r"\[\s*\d+\s*,\s*\d+"), "text contains a token id list"),
    (re.compile(r"TranscriptionResult|object at 0x"), "text contains a repr of an object"),
]

# Доля кириллицы среди букв. Модели multilingual вставляют английские куски,
# поэтому порог невысокий - ловим именно мусор, а не смешанную речь.
MIN_CYRILLIC_RATIO = 0.5


def check_text(text: str) -> list[str]:
    """Проверяет текст транскрипции, возвращает список проблем."""
    problems = []

    if not text or not text.strip():
        return ["empty text"]

    if len(text) < MIN_TEXT_LENGTH:
        problems.append(f"text too short: {len(text)} chars < {MIN_TEXT_LENGTH}")

    for pattern, message in LEAK_PATTERNS:
        if pattern.search(text):
            problems.append(f"{message}: {text[:80]!r}")

    lowered = text.lower()
    missing = [s for s in EXPECTED_SUBSTRINGS if s not in lowered]
    if missing:
        problems.append(f"expected {missing} in text, got: {text[:80]!r}")

    letters = [c for c in text if c.isalpha()]
    if letters:
        cyrillic = sum(1 for c in letters if "Ѐ" <= c <= "ӿ")
        ratio = cyrillic / len(letters)
        if ratio < MIN_CYRILLIC_RATIO:
            problems.append(f"cyrillic ratio {ratio:.2f} < {MIN_CYRILLIC_RATIO}")

    return problems


def check_response(result: dict) -> list[str]:
    """Проверяет структуру ответа API."""
    problems = check_text(result.get("text") or "")

    segments = result.get("segments")
    if segments:
        for i, seg in enumerate(segments):
            seg_text = seg.get("text") or ""
            for pattern, message in LEAK_PATTERNS:
                if pattern.search(seg_text):
                    problems.append(f"segment {i}: {message}")
                    break
            start, end = seg.get("start"), seg.get("end")
            if start is not None and end is not None and end < start:
                problems.append(f"segment {i}: end {end} < start {start}")

    return problems


def run_engine(entry: dict, audio_path: Path) -> list[str]:
    """Поднимает движок, транскрибирует, проверяет результат."""
    name, script = entry["name"], entry["script"]
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")

    if not (PROJECT_ROOT / script).exists():
        return [f"script not found: {script}"]

    process = None
    try:
        process = start_server(script, entry["env"])
        if process is None:
            return ["failed to start server"]

        if not wait_for_server(process=process):
            if process.poll() is not None:
                out = ""
                try:
                    out = process.stdout.read().decode("utf-8", "replace")[-400:]
                except Exception:
                    pass
                return [f"server exited (code {process.returncode}): {out}"]
            return ["server startup timeout"]

        result = transcribe_audio(audio_path)
        text = result.get("text") or ""
        print(f"  text[:100]: {text[:100]!r}")
        print(f"  length: {len(text)} chars, segments: {len(result.get('segments') or [])}")

        problems = check_response(result)
        if problems:
            for p in problems:
                print(f"  FAIL: {p}")
        else:
            print("  OK")
        return problems

    except Exception as e:
        return [f"{type(e).__name__}: {e}"]
    finally:
        if process is not None:
            reset_session()
            stop_server(process)
            if not wait_for_port_free(SERVER_PORT, timeout=10):
                kill_process_on_port(SERVER_PORT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ASR engine output")
    parser.add_argument(
        "--engine", "-e", action="append", default=None,
        help="Run only these engines (by name or script). Repeatable."
    )
    args = parser.parse_args()

    entries = BENCHMARK_SCRIPTS
    if args.engine:
        wanted = set(args.engine)
        entries = [e for e in BENCHMARK_SCRIPTS
                   if e["name"] in wanted or e["script"] in wanted]
        if not entries:
            print(f"No engines matching: {args.engine}")
            return 2

    audio_path = get_test_audio()
    print(f"Test audio: {audio_path.name}")

    if is_port_in_use(SERVER_PORT):
        kill_process_on_port(SERVER_PORT)
        wait_for_port_free(SERVER_PORT, timeout=10)

    failures = {}
    for entry in entries:
        problems = run_engine(entry, audio_path)
        if problems:
            failures[entry["name"]] = problems

    print(f"\n{'=' * 60}")
    if failures:
        print(f"FAILED: {len(failures)}/{len(entries)} engines")
        for name, problems in failures.items():
            print(f"  {name}:")
            for p in problems:
                print(f"    - {p}")
        return 1

    print(f"PASSED: all {len(entries)} engines produce valid output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
