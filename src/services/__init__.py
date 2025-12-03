"""
OAITT — Open AI Transformer Transcriber.

Сервисные модули.

Содержит:
- performance: Отслеживание производительности транскрипции
- timeout: Управление таймаутами транскрипции
- debug: Отладочное логирование аудио и результатов

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

from src.services.performance import PerformanceTracker
from src.services.timeout import transcribe_with_timeout, TranscriptionTimeoutError
from src.services.debug import save_debug_log

__all__ = [
    "PerformanceTracker",
    "transcribe_with_timeout",
    "TranscriptionTimeoutError",
    "save_debug_log",
]
