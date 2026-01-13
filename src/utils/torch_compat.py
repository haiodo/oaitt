"""
OAITT — Open AI Transformer Transcriber.

PyTorch compatibility utilities.
Provides workarounds for PyTorch 2.6+ weights_only loading issues
that affect libraries like pyannote.audio, whisperx, and others
using older checkpoint formats.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import logging
from functools import wraps

logger = logging.getLogger(__name__)

_torch_load_patched = False


def patch_torch_load_weights_only():
    """
    Patch torch.load to use weights_only=False by default.

    This is a workaround for PyTorch 2.6+ which changed the default
    value of weights_only from False to True. Many older model checkpoints
    (e.g., pyannote.audio VAD models used by WhisperX) contain objects
    that require pickle unpickling and fail with weights_only=True.

    This patch should be called BEFORE importing libraries that load
    such checkpoints (e.g., whisperx, pyannote.audio).

    Note:
        This reduces security as it allows arbitrary code execution
        during model loading. Only use with trusted model sources.

    Example:
        >>> from src.utils.torch_compat import patch_torch_load_weights_only
        >>> patch_torch_load_weights_only()
        >>> import whisperx  # Now whisperx can load VAD models
    """
    global _torch_load_patched

    if _torch_load_patched:
        logger.debug("torch.load already patched, skipping")
        return

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, cannot patch torch.load")
        return

    # Check if we're on PyTorch 2.6+ where this is needed
    torch_version = torch.__version__.split('+')[0]  # Remove cuda suffix if present
    version_parts = torch_version.split('.')
    try:
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    except (ValueError, IndexError):
        major, minor = 2, 6  # Assume recent version

    if major < 2 or (major == 2 and minor < 6):
        logger.debug(
            f"PyTorch version {torch_version} doesn't require weights_only patch"
        )
        return

    # Store original function
    _original_torch_load = torch.load

    @wraps(_original_torch_load)
    def _patched_torch_load(*args, **kwargs):
        """
        Patched torch.load that forces weights_only=False.

        This allows loading older checkpoints that contain non-tensor objects
        like omegaconf configs, which are common in audio/speech models.
        """
        # Force weights_only=False to allow unpickling of arbitrary objects
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)

    # Apply patch
    torch.load = _patched_torch_load
    _torch_load_patched = True

    logger.info(
        f"Patched torch.load for PyTorch {torch_version} compatibility "
        "(weights_only=False)"
    )


def add_omegaconf_safe_globals():
    """
    Add omegaconf types to torch's safe globals list.

    This is an alternative to patching torch.load that maintains
    weights_only=True but allows specific omegaconf types to be
    unpickled safely.

    Note:
        This approach is more secure than patching torch.load but
        may not cover all types used by various models.
    """
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not installed, cannot add safe globals")
        return

    try:
        from omegaconf import ListConfig, DictConfig
        from omegaconf.base import ContainerMetadata

        torch.serialization.add_safe_globals([
            ListConfig,
            DictConfig,
            ContainerMetadata,
        ])
        logger.info("Added omegaconf types to torch safe globals")
    except ImportError:
        logger.debug("omegaconf not installed, skipping safe globals setup")
    except Exception as e:
        logger.warning(f"Failed to add omegaconf safe globals: {e}")


def ensure_torch_compatibility():
    """
    Ensure PyTorch compatibility for loading older model checkpoints.

    This is a convenience function that applies necessary patches
    for compatibility with libraries like WhisperX that use older
    checkpoint formats.

    Should be called early in the application startup, before
    importing libraries that load model checkpoints.

    Example:
        >>> from src.utils.torch_compat import ensure_torch_compatibility
        >>> ensure_torch_compatibility()
        >>> import whisperx
        >>> model = whisperx.load_model(...)
    """
    patch_torch_load_weights_only()
