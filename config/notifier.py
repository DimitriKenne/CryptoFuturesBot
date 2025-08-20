# config/notifier_config_schema.py

from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class TelegramConfig:
    """Telegram notification settings."""
    enabled: bool = True
    token: str = field(default_factory=lambda: os.getenv('TELEGRAM_BOT_TOKEN', ''))
    chat_id: str = field(default_factory=lambda: os.getenv('TELEGRAM_CHAT_ID', ''))

@dataclass
class NotifierConfig:
    """Notification service settings."""
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

# Default config instance
DEFAULT_NOTIFIER_CONFIG = NotifierConfig()
