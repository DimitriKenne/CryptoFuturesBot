# utils/notification_manager.py

import logging
import requests # For simple webhook/API calls like Telegram
from typing import Dict, Optional
import sys # Import sys
import asyncio # Import asyncio for async operations

# Get the logger for this module
logger = logging.getLogger(__name__)

# Optional: Try importing telegram library, but handle if not installed
try:
    import telegram
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    logger.warning("python-telegram-bot library not found. Telegram notifications disabled. "
                   "Install with: pip install python-telegram-bot")

class NotificationManager:
    """
    Handles sending notifications for trading bot events via logging and optionally Telegram.
    Extendable for other services (Email, Slack).
    """
    def __init__(self, config: Optional[Dict] = None):
        """
        Initializes the notification manager.

        Args:
            config (Optional[Dict]): Configuration dictionary potentially containing:
                - telegram (Dict): Telegram specific configuration
                    - enabled (bool): Whether to send Telegram messages.
                    - token (str): Your Telegram Bot Token.
                    - chat_id (str): The target chat ID for notifications.
                - slack (Dict): Slack specific configuration (example)
                    - webhook_url (str): Your Slack incoming webhook URL.
                - Other service-specific keys...
        """
        self.config = config if config else {}

        # --- Telegram Configuration ---
        telegram_config = self.config.get('telegram', {})
        self.telegram_enabled = telegram_config.get('enabled', False)
        self.telegram_token = telegram_config.get('token')
        self.telegram_chat_id = telegram_config.get('chat_id')
        self.telegram_bot = None # Initialize bot instance

        # Validate Telegram config if enabled
        if self.telegram_enabled:
            if not TELEGRAM_AVAILABLE:
                logger.error("Telegram notifications enabled in config, but 'python-telegram-bot' library is not installed. Disabling Telegram.")
                self.telegram_enabled = False
            elif not self.telegram_token:
                logger.error("Telegram notifications enabled, but 'token' is missing in telegram config. Disabling Telegram.")
                self.telegram_enabled = False
            elif not self.telegram_chat_id:
                logger.error("Telegram notifications enabled, but 'chat_id' is missing in telegram config. Disabling Telegram.")
                self.telegram_enabled = False
            else:
                # Initialize the bot instance if enabled and config is valid
                try:
                    self.telegram_bot = telegram.Bot(token=self.telegram_token)
                    logger.info("Telegram notifications enabled.")
                except Exception as e:
                    logger.error(f"Failed to initialize Telegram bot: {e}. Disabling Telegram notifications.", exc_info=True)
                    self.telegram_enabled = False
                    self.telegram_bot = None # Ensure bot is None if initialization fails

        else:
            logger.info("Telegram notifications disabled.")


        # --- Slack Configuration (Example - Uncomment and implement _send_slack_message if needed) ---
        # slack_config = self.config.get('slack', {})
        # self.slack_webhook_url = slack_config.get('webhook_url')
        # if self.slack_webhook_url:
        #     logger.info("Slack notifications enabled.")
        # else:
        #     logger.info("Slack notifications disabled.")


        logger.info("NotificationManager initialized.")

    async def _send_telegram_message(self, message: str):
        """Sends a message using the python-telegram-bot library (async)."""
        if not self.telegram_enabled or not TELEGRAM_AVAILABLE or self.telegram_bot is None:
            return # Skip if disabled, library missing, or bot not initialized

        try:
            # Use the initialized bot instance
            # Await the asynchronous send_message method
            await self.telegram_bot.send_message(chat_id=self.telegram_chat_id, text=message)
            logger.debug(f"Telegram notification sent successfully to chat ID {self.telegram_chat_id}.")
        except telegram.error.TelegramError as e:
            logger.error(f"Telegram API error sending notification: {e}")
            # Potentially disable Telegram temporarily if errors persist
        except Exception as e:
            logger.error(f"Failed to send Telegram notification due to unexpected error: {e}", exc_info=True)

    # This method can remain synchronous if it only uses synchronous requests (like the example)
    def _send_slack_message(self, message: str, level: str):
        """Sends a message using a Slack incoming webhook."""
        slack_webhook_url = self.config.get('slack', {}).get('webhook_url') # Get from nested config
        if not slack_webhook_url:
            return # Skip if not configured

        try:
            payload = {'text': f"[{level.upper()}] {message}"}
            # requests.post is typically synchronous, so no await needed here
            response = requests.post(slack_webhook_url, json=payload, timeout=10)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            logger.debug("Slack notification sent successfully.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")
        except Exception as e:
            logger.error(f"Unexpected error sending Slack notification: {e}", exc_info=True)


    async def send_notification(self, message: str, level: str = "info"):
        """
        Sends a notification via configured channels (logging, Telegram, etc.) (async).

        Args:
            message (str): The message content.
            level (str): The severity level ('info', 'warning', 'error', 'critical').
                         Used for logging level and potentially message formatting.
        """
        level = level.lower() # Ensure lowercase for comparisons
        log_message = f"NOTIFICATION: {message}" # Keep log message clean

        # --- 1. Logging (Always Active) ---
        if level == "info":
            logger.info(log_message)
        elif level == "warning":
            logger.warning(log_message)
        elif level == "error":
            logger.error(log_message)
        elif level == "critical":
            logger.critical(log_message)
        else:
            logger.info(f"NOTIFICATION (level={level}): {message}") # Log with level if unknown

        # --- 2. Telegram ---
        if self.telegram_enabled:
            # Get symbol from a higher level config if available, or default
            # Assuming SYMBOL might be passed in the config dict during initialization
            symbol = self.config.get('SYMBOL', '')
            telegram_msg = f"[{level.upper()}] {symbol} Bot: {message}" if symbol else f"[{level.upper()}] Bot: {message}"
            # Await the asynchronous Telegram sending method
            await self._send_telegram_message(telegram_msg)

        # --- 3. Slack ---
        # self._send_slack_message(message, level) # Uncomment if Slack is implemented

        # --- 4. Other Services (Email, etc.) ---
        # Add calls to other sender methods here based on config flags

