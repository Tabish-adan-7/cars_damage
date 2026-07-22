"""
Configuration Module
Centralized configuration management for production deployment

Handles:
- Environment variables
- Security (API key protection)
- Paths and directories
- Model settings
- Default values
"""

import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Configuration class for the car damage detection system.

    All sensitive data (API keys) should be in environment variables.
    Configuration is validated on initialization.
    """

    # ==================== PATHS ====================
    BASE_DIR = Path(__file__).parent
    TEMP_DIR = BASE_DIR / "temp_images"
    LOG_DIR = BASE_DIR / "logs"
    MODELS_DIR = BASE_DIR / "models"

    # ==================== MODEL CONFIGURATION ====================
    # YOLO Model
    YOLO_MODEL_PATH = os.getenv(
        "YOLO_MODEL_PATH",
        str(BASE_DIR / "best.pt")
    )

    # Groq Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Alternative models available in Groq
    AVAILABLE_GROQ_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen-3-32b",
    ]

    # ==================== APPLICATION SETTINGS ====================
    # UI Settings
    APP_TITLE = "Car Damage Detection & AI Assistant"
    APP_VERSION = "1.0.0"

    # Detection Settings
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    MIN_CONFIDENCE_THRESHOLD = 0.0
    MAX_CONFIDENCE_THRESHOLD = 1.0

    # Image Settings
    MAX_IMAGE_SIZE_MB = 20
    ALLOWED_IMAGE_FORMATS = ("jpg", "jpeg", "png")
    IMAGE_QUALITY = 95

    # AI Settings
    AI_MAX_RESPONSE_LENGTH = 500
    AI_TEMPERATURE = 0.7
    AI_TOP_P = 0.9
    CONVERSATION_HISTORY_LIMIT = 20

    # Performance
    USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"

    # ==================== LOGGING ====================
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE = os.getenv("LOG_FILE", str(LOG_DIR / "car_damage_detection.log"))

    # ==================== SECURITY ====================
    # Data retention
    TEMP_FILE_RETENTION_HOURS = 1  # Delete temp images after 1 hour

    # Privacy
    STORE_IMAGES_PERMANENTLY = False
    STORE_CONVERSATION_LOGS = True
    ANONYMIZE_LOGS = True

    def __init__(self):
        """Initialize and validate configuration."""
        self._validate_config()
        self._create_directories()
        logger.info("Configuration loaded and validated")

    def _validate_config(self):
        """Validate required configuration parameters."""
        errors = []

        # Check API keys
        if not self.GROQ_API_KEY:
            errors.append("GROQ_API_KEY environment variable is not set")

        # Check model path
        if not Path(self.YOLO_MODEL_PATH).exists():
            errors.append(
                f"YOLO model not found at {self.YOLO_MODEL_PATH}. "
                f"Set YOLO_MODEL_PATH environment variable."
            )

        # Validate model choice
        if self.GROQ_MODEL not in self.AVAILABLE_GROQ_MODELS:
            logger.warning(
                f"Model '{self.GROQ_MODEL}' not in standard list. "
                f"Will attempt to use it anyway."
            )

        if errors:
            logger.error("Configuration validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("Configuration validation failed. See logs.")

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.TEMP_DIR, self.LOG_DIR, self.MODELS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ready: {directory}")

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"

    @property
    def debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return os.getenv("DEBUG", "false").lower() == "true"

    def get_model_config(self) -> dict:
        """Get YOLO model configuration."""
        return {
            "path": self.YOLO_MODEL_PATH,
            "confidence": self.DEFAULT_CONFIDENCE_THRESHOLD,
            "use_gpu": self.USE_GPU,
        }

    def get_ai_config(self) -> dict:
        """Get AI agent configuration."""
        return {
            "api_key": self.GROQ_API_KEY,
            "model": self.GROQ_MODEL,
            "temperature": self.AI_TEMPERATURE,
            "max_tokens": self.AI_MAX_RESPONSE_LENGTH,
            "top_p": self.AI_TOP_P,
        }

    def get_image_config(self) -> dict:
        """Get image processing configuration."""
        return {
            "max_size_mb": self.MAX_IMAGE_SIZE_MB,
            "allowed_formats": self.ALLOWED_IMAGE_FORMATS,
            "quality": self.IMAGE_QUALITY,
        }

    @staticmethod
    def from_env(**kwargs) -> 'Config':
        """
        Create config from environment with optional overrides.

        Args:
            **kwargs: Override specific config values

        Returns:
            Config instance
        """
        config = Config()

        # Apply overrides if provided
        for key, value in kwargs.items():
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), value)

        return config


# Singleton instance (optional)
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get or create singleton config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance