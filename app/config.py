"""
Configuration settings for Facteur Tabimetrique API.

This module provides centralized configuration using Pydantic Settings,
supporting environment variables and .env file.

Author: EYAGA TABI Jean François Régis
Contact: francoistabi294@gmail.com
GitHub: https://github.com/vulgane034
LinkedIn: https://www.linkedin.com/in/francois-tabi-03a4b7235
"""

import logging
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API configuration using Pydantic Settings.
    
    Configuration can be set via:
    - Environment variables
    - .env file
    - Direct class attributes
    
    Attributes:
        api_title: API title
        api_version: API version
        api_host: API host address
        api_port: API port number
        cors_origins: List of allowed CORS origins
        max_models_in_memory: Maximum number of models to keep in memory
        debug: Debug mode flag
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        mlp_epochs: MLP training epochs
        mlp_batch_size: MLP batch size
        mlp_validation_split: MLP validation split ratio
        mlp_learning_rate: MLP learning rate
        max_upload_size: Maximum upload file size in MB
    """

    # API Configuration
    api_title: str = "Facteur Tabimetrique API"
    api_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # CORS Configuration
    cors_origins: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    
    # Model Storage
    max_models_in_memory: int = 50
    model_storage_limit: int = 50  # Alias for compatibility
    
    # Logging & Debug
    debug: bool = False
    log_level: str = "INFO"
    
    # MLP Training Configuration
    mlp_epochs: int = 100
    mlp_batch_size: int = 32
    mlp_validation_split: float = 0.2
    mlp_learning_rate: float = 0.001
    
    # File Upload
    max_upload_size: int = 100  # MB

    class Config:
        """Pydantic settings configuration.
        
        Attributes:
            env_file: Path to .env file
            case_sensitive: Whether environment variables are case-sensitive
            env_file_encoding: Encoding for .env file
        """
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def setup_logging(level: str = "INFO") -> None:
    """Configure application logging.
    
    Sets up both console and file logging with standardized format.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Raises:
        None
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("app.log", encoding="utf-8"),
        ],
    )


# Global settings instance
settings = Settings()

# Setup logging
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)

# Log configuration on startup
logger.info(f"API: {settings.api_title} v{settings.api_version}")
logger.info(f"Server: {settings.api_host}:{settings.api_port}")
logger.info(f"Debug mode: {settings.debug}")
logger.info(f"Models limit: {settings.max_models_in_memory}")
logger.info(f"CORS origins: {settings.cors_origins}")
