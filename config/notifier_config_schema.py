# config/notifier_config_schema.py

from dataclasses import dataclass, field
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class TelegramConfig:
    """Configuration for Telegram notifications."""
    enabled: bool = False
    token: str = field(default_factory=lambda: os.getenv('TELEGRAM_BOT_TOKEN', ''))
    chat_id: str = field(default_factory=lambda: os.getenv('TELEGRAM_CHAT_ID', ''))

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("Telegram 'enabled' must be a boolean.")
        if self.enabled:
            if not isinstance(self.token, str) or not self.token:
                logger.warning("Telegram notifications enabled, but 'token' is missing. Notifications will likely fail.")
            if not isinstance(self.chat_id, str) or not self.chat_id:
                logger.warning("Telegram notifications enabled, but 'chat_id' is missing. Notifications will likely fail.")

@dataclass
class NotifierConfig:
    """
    Defines configuration parameters for various notification services.
    """
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    def __post_init__(self):
        if isinstance(self.telegram, dict):
            self.telegram = TelegramConfig(**self.telegram)
        elif not isinstance(self.telegram, TelegramConfig):
            raise TypeError("telegram config must be a dictionary or TelegramConfig instance.")

# Default configuration instance
DEFAULT_NOTIFIER_CONFIG = NotifierConfig()
