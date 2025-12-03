"""
OAITT — Open AI Transformer Transcriber.

Главный модуль приложения FastAPI.
Создаёт и настраивает экземпляр FastAPI приложения,
регистрирует маршруты и обработчики событий жизненного цикла.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src import __version__
from src.asr.factory import create_asr_model
from src.asr.base import ASRModel
from src.config import ASR_ENGINE, HOST, PORT
from src.routes import asr_router, health_router, openai_router
from src.routes.asr import set_asr_model as set_asr_model_asr
from src.routes.health import set_asr_model as set_asr_model_health
from src.routes.openai import set_asr_model as set_asr_model_openai

logger = logging.getLogger(__name__)

# Global ASR model instance
_asr_model: ASRModel | None = None


def get_asr_model() -> ASRModel | None:
    """
    Возвращает глобальный экземпляр ASR модели.

    Returns:
        Экземпляр ASR модели или None, если модель не инициализирована.
    """
    return _asr_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управляет жизненным циклом приложения.

    Загружает модель при запуске и освобождает ресурсы при остановке.

    Args:
        app: Экземпляр FastAPI приложения.
    """
    global _asr_model

    # Startup
    logger.info(f"Starting OAITT (Open AI Transformer Transcriber) v{__version__}")
    logger.info(f"Initializing ASR with engine: {ASR_ENGINE}")

    try:
        _asr_model = create_asr_model()
        _asr_model.load_model()

        # Set model reference in all routers
        set_asr_model_asr(_asr_model)
        set_asr_model_health(_asr_model)
        set_asr_model_openai(_asr_model)

        logger.info("ASR model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down OAITT")
    if _asr_model is not None:
        _asr_model.release_model()
        logger.info("ASR model released")


def create_app() -> FastAPI:
    """
    Создаёт и настраивает экземпляр FastAPI приложения.

    Returns:
        FastAPI: Настроенное приложение.

    Example:
        >>> app = create_app()
        >>> # Run with uvicorn
        >>> import uvicorn
        >>> uvicorn.run(app, host="0.0.0.0", port=9007)
    """
    app = FastAPI(
        title="OAITT — Open AI Transformer Transcriber",
        description=(
            "Speech-to-text transcription service powered by OpenAI Whisper models.\n\n"
            "Supports two ASR engines:\n"
            "- **transformers**: Hugging Face Transformers pipeline with Whisper models\n"
            "- **whisperx**: WhisperX with word-level alignment\n\n"
            "Provides OpenAI-compatible API endpoint for easy integration."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(asr_router)
    app.include_router(openai_router)

    return app


# Create the app instance
app = create_app()


def run_server(host: str | None = None, port: int | None = None) -> None:
    """
    Запускает сервер с помощью uvicorn.

    Args:
        host: Хост для привязки (по умолчанию из конфигурации).
        port: Порт для привязки (по умолчанию из конфигурации).

    Example:
        >>> from src.app import run_server
        >>> run_server(host="0.0.0.0", port=9007)
    """
    import uvicorn

    uvicorn.run(
        app,
        host=host or HOST,
        port=port or PORT,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
