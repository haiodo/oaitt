"""
OAITT — Open AI Transformer Transcriber.

Конфигурация приложения.
Все настройки загружаются из переменных окружения.

Copyright (c) 2025 Andrey Sobolev (haiodo@gmail.com)
Licensed under MIT License.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ASR Engine Configuration
# =============================================================================

# ASR engine to use: "transformers" or "whisperx"
ASR_ENGINE = os.getenv("ASR_ENGINE", "transformers")

# Model name for Hugging Face Transformers engine
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3")

# Model name for WhisperX engine
WHISPERX_MODEL = os.getenv("WHISPERX_MODEL", "large-v3")

# Device to use: "auto", "cuda", "cpu", "mps"
DEVICE = os.getenv("DEVICE", "auto")

# Compute type for WhisperX: "float16", "float32", "int8", etc.
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float32")

# Seconds before unloading idle model (0 = never)
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", "0"))

# Directory for caching downloaded models
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./data")

# Directory for saving audio chunks and results for debugging (None = disabled)
DEBUG_LOG_DIR = os.getenv("DEBUG_LOG_DIR", None)

# Audio sample rate (Whisper requirement)
SAMPLE_RATE = 16000

# =============================================================================
# Server Configuration
# =============================================================================

# Host to bind the server
HOST = os.getenv("HOST", "0.0.0.0")

# Port to bind the server
PORT = int(os.getenv("PORT", "9007"))

# =============================================================================
# Adaptive Timeout Configuration
# =============================================================================

# Enable/disable adaptive timeout
TIMEOUT_ENABLED = os.getenv("TIMEOUT_ENABLED", "false").lower() == "true"

# Multiplier for expected processing time before timeout (e.g., 2.0 = allow 2x average time)
TIMEOUT_MULTIPLIER = float(os.getenv("TIMEOUT_MULTIPLIER", "2.0"))

# Minimum timeout in seconds (fallback when no history available)
TIMEOUT_MIN_SECONDS = float(os.getenv("TIMEOUT_MIN_SECONDS", "30.0"))

# Maximum timeout in seconds (safety cap)
TIMEOUT_MAX_SECONDS = float(os.getenv("TIMEOUT_MAX_SECONDS", "300.0"))

# Number of recent samples to keep for average calculation
TIMEOUT_HISTORY_SIZE = int(os.getenv("TIMEOUT_HISTORY_SIZE", "100"))

# =============================================================================
# Confidence Thresholds Configuration
# =============================================================================

# avg_logprob: average log probability of tokens (higher is better, typically -0.5 to 0)
CONFIDENCE_AVG_LOGPROB_THRESHOLD = float(os.getenv("CONFIDENCE_AVG_LOGPROB_THRESHOLD", "-1.0"))

# no_speech_prob: probability that segment contains no speech (lower is better, 0 to 1)
CONFIDENCE_NO_SPEECH_THRESHOLD = float(os.getenv("CONFIDENCE_NO_SPEECH_THRESHOLD", "0.6"))

# word_score: minimum average word alignment score (higher is better, 0 to 1)
CONFIDENCE_WORD_SCORE_THRESHOLD = float(os.getenv("CONFIDENCE_WORD_SCORE_THRESHOLD", "0.5"))

# word_prob: minimum average word probability from model (higher is better, 0 to 1)
CONFIDENCE_WORD_PROB_THRESHOLD = float(os.getenv("CONFIDENCE_WORD_PROB_THRESHOLD", "0.4"))

# low_prob_word_ratio: maximum ratio of low-probability words allowed (0 to 1)
CONFIDENCE_LOW_PROB_RATIO_THRESHOLD = float(os.getenv("CONFIDENCE_LOW_PROB_RATIO_THRESHOLD", "0.5"))

# Enable/disable automatic filtering of low-confidence results
CONFIDENCE_FILTER_ENABLED = os.getenv("CONFIDENCE_FILTER_ENABLED", "false").lower() == "true"

# =============================================================================
# Characters-per-second (chars/sec) Configuration
# =============================================================================
# Baseline characters per second considered normal (approx 20-30 chars/sec typical speaking rate)
MAX_CHARS_PER_SECOND = float(os.getenv("MAX_CHARS_PER_SECOND", "25.0"))

# Multiplier - if observed chars/sec exceeds baseline * multiplier, mark as suspicious
CHARS_PER_SECOND_MULTIPLIER = float(os.getenv("CHARS_PER_SECOND_MULTIPLIER", "3.0"))

# Minimum audio duration (seconds) to apply chars/sec checks (avoid noisy short-audio edge cases)
CHARS_PER_SECOND_MIN_AUDIO_SEC = float(os.getenv("CHARS_PER_SECOND_MIN_AUDIO_SEC", "0.5"))


# =============================================================================
# Authentication Configuration
# =============================================================================

# API token for authentication (set to empty string to disable)
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "key")

# =============================================================================
# Initialize Cache Directories
# =============================================================================

def init_cache_directories():
    """Set cache directories if specified."""
    if MODEL_CACHE_DIR:
        os.environ["HF_HOME"] = MODEL_CACHE_DIR
        os.environ["TORCH_HOME"] = os.path.join(MODEL_CACHE_DIR, "torch")

    if DEBUG_LOG_DIR:
        os.makedirs(DEBUG_LOG_DIR, exist_ok=True)
        logger.info(f"Debug logging enabled, saving to: {DEBUG_LOG_DIR}")


# Initialize on import
init_cache_directories()
