#!/usr/bin/env python3
"""
OAITT — Open AI Transformer Transcriber.

Точка входа для запуска сервиса распознавания речи.
Использует настроенный ASR движок (transformers или whisperx).

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.

Usage:
    python main.py

Environment Variables:
    ASR_ENGINE: ASR engine to use (transformers, whisperx)
    HOST: Host to bind the server (default: 0.0.0.0)
    PORT: Port to bind the server (default: 9007)
    DEVICE: Device to use (auto, cuda, cpu, mps)

See README.md for full configuration options.
"""

import logging
import sys

# Configure logging before importing other modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def setup_torch_serialization():
    """
    Настраивает безопасные глобальные переменные для torch.load.

    Необходимо для загрузки моделей pyannote и других библиотек,
    которые сохраняют объекты с кастомными классами.
    """
    import collections
    import typing

    import torch
    import omegaconf
    import pyannote.audio

    # Add safe globals for torch.load
    safe_globals = [
        omegaconf.listconfig.ListConfig,
        omegaconf.base.ContainerMetadata,
        omegaconf.nodes.AnyNode,
        omegaconf.base.Metadata,
        typing.Any,
        list,
        dict,
        int,
        collections.defaultdict,
        torch.torch_version.TorchVersion,
        pyannote.audio.core.model.Introspection,
        pyannote.audio.core.task.Specifications,
        pyannote.audio.core.task.Problem,
        pyannote.audio.core.task.Resolution,
    ]

    for cls in safe_globals:
        torch.serialization.add_safe_globals([cls])


def main():
    """Главная функция запуска сервиса."""
    logger.info("=" * 60)
    logger.info("OAITT — Open AI Transformer Transcriber")
    logger.info("=" * 60)

    # Import config to trigger cache directory initialization
    from src.config import ASR_ENGINE, HOST, PORT, DEVICE

    logger.info(f"Configuration:")
    logger.info(f"  ASR Engine: {ASR_ENGINE}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Server: {HOST}:{PORT}")
    logger.info("=" * 60)

    # Setup torch serialization for safe model loading
    try:
        setup_torch_serialization()
    except Exception as e:
        logger.warning(f"Failed to setup torch serialization: {e}")

    # Import and run the application
    try:
        from src.app import run_server
        run_server()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
