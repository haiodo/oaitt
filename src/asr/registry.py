"""
OAITT — Open AI Transformer Transcriber.

Реестр моделей: держит несколько ASR-движков в одном процессе и выбирает нужный
по полю `model` из запроса.

Один процесс с двумя моделями удобнее двух деплоев: CTC там, где важна скорость,
RNNT там, где важно качество, и переключение делает клиент, а не админ. Цена -
вторая копия весов в unified memory (около 850 МБ на модель GigaAM).

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
import threading
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from src.asr.base import ASRModel

logger = logging.getLogger(__name__)

# Имя модели -> (движок, аргумент конструктора). Имя - то, что клиент шлёт в поле `model`.
KNOWN_MODELS: Dict[str, tuple] = {
    "gigaam-ctc": ("gigaam_mlx", "ctc"),
    "gigaam-rnnt": ("gigaam_mlx", "rnnt"),
    "gigaam-multilingual-int8": ("gigaam_multilingual_mlx", "int8"),
    "gigaam-multilingual-fp16": ("gigaam_multilingual_mlx", "fp16"),
}


def _build(engine: str, argument: Optional[str]) -> "ASRModel":
    if engine == "gigaam_mlx":
        from src.asr.gigaam_mlx import GigaAMMLXASR

        return GigaAMMLXASR(model_type=argument)

    if engine == "gigaam_multilingual_mlx":
        from src.asr.gigaam_multilingual_mlx import GigaAMMultilingualMLXASR

        return GigaAMMultilingualMLXASR(variant=argument)

    from src.asr.factory import create_asr_model

    return create_asr_model()


class ModelRegistry:
    """
    Ленивый реестр моделей.

    Модель грузится при первом обращении к ней, а не на старте: держать в памяти
    только то, что реально спрашивают.

    Args:
        names: имена из `KNOWN_MODELS`, которые разрешено поднимать.
        default: модель для запросов без `model` или с незнакомым именем.
            Клиенты присылают и `whisper-1`, и `gigaam` - ронять их из-за этого нельзя.

    Example:
        >>> registry = ModelRegistry(["gigaam-ctc", "gigaam-rnnt"], default_model=default)
        >>> model = registry.get("gigaam-rnnt")
    """

    def __init__(self, names: List[str], default_model: "ASRModel"):
        self._names = [n for n in names if n in KNOWN_MODELS]
        self._default = default_model
        self._models: Dict[str, "ASRModel"] = {}
        self._lock = threading.Lock()

        unknown = [n for n in names if n not in KNOWN_MODELS]
        if unknown:
            logger.warning(f"Unknown models in ASR_MODELS, ignored: {', '.join(unknown)}")
        logger.info(f"Model registry: {', '.join(self._names) or '(default only)'}")

    @property
    def names(self) -> List[str]:
        return list(self._names)

    def get(self, name: Optional[str]) -> "ASRModel":
        """Модель по имени; неизвестное имя - дефолтная."""
        key = (name or "").strip().lower()
        if key not in self._names:
            return self._default

        with self._lock:
            model = self._models.get(key)
            if model is None:
                engine, argument = KNOWN_MODELS[key]
                logger.info(f"Loading model '{key}' ({engine}, {argument})")
                model = _build(engine, argument)
                model.ensure_model_loaded()
                self._models[key] = model
        return model

    def loaded(self) -> List[str]:
        return sorted(self._models)

    def release_all(self) -> None:
        with self._lock:
            for name, model in self._models.items():
                logger.info(f"Releasing model '{name}'")
                model.release_model()
            self._models.clear()
