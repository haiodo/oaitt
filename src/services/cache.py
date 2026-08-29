"""
OAITT — Open AI Transformer Transcriber.

Кеш результатов транскрипции по содержимому аудио.

Платформа ретраит задачи: до 5 попыток на transient-ошибках и без ограничения на
сетевых (services/ai-bot/pod-ai-bot/src/transcription/consumer.ts). Каждый повтор
приносит тот же чанк, и без кеша он считается заново. Ключ - хеш самого аудио плюс
параметры, поэтому совпадение означает буквально тот же запрос.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def audio_key(audio: np.ndarray, **params: Any) -> str:
    """Ключ по содержимому аудио и параметрам запроса."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(audio, dtype=np.float32).tobytes())
    for name in sorted(params):
        digest.update(f"|{name}={params[name]}".encode())
    return digest.hexdigest()


class ResultCache:
    """
    LRU с TTL. Не потокобезопасен снаружи - все операции под своим локом.

    Args:
        max_entries: сколько результатов держать; 0 выключает кеш.
        ttl_sec: сколько запись живёт. Ретраи платформы приходят в пределах минут,
            дольше держать смысла нет - аудио уже не повторится.
    """

    def __init__(self, max_entries: int = 256, ttl_sec: float = 300.0):
        self._max_entries = max_entries
        self._ttl = ttl_sec
        self._entries: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._max_entries > 0 and self._ttl > 0

    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if time.time() - stored_at > self._ttl:
                del self._entries[key]
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[key] = (time.time(), value)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "enabled": self.enabled,
                "entries": len(self._entries),
                "max_entries": self._max_entries,
                "ttl_sec": self._ttl,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / total, 4) if total else None,
            }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
