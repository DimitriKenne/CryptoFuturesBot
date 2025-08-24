# utils/strategy_execution/live_trading_session_manager.py

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, timezone

from config.params import AppConfig

logger = logging.getLogger(__name__)

class LiveTradingSessionManager:
    """
    Manages the state for a LIVE trading session, including capital, positions,
    trade history, and rules specific to live trading. This class is designed for
    state persistence and recovery.
    """

    def __init__(self, app_config: AppConfig):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing LiveTradingSessionManager...")

        self.config = app_config
        self.trading_config = app_config.trading
        self.general_config = app_config.general

        # Internal state variables
        self._current_capital: float = float(self.trading_config.risk.initial_capital)
        self._open_position: Optional[Dict[str, Any]] = None
        self._trade_history: List[Dict[str, Any]] = []
        self._last_processed_timestamp: Optional[datetime] = None
        self._last_trade_timestamp: Optional[datetime] = None

        self.logger.info(f"LiveTradingSessionManager initialized with initial capital: {self._current_capital:.2f}")

    def get_current_capital(self) -> float:
        """Returns the current cash balance."""
        return self._current_capital

    def get_open_position(self) -> Optional[Dict[str, Any]]:
        """Returns the details of the currently open position, or None if flat."""
        return self._open_position

    def set_open_position(self, position: Dict[str, Any]):
        """Sets the current open position and records its timestamp."""
        self._open_position = position
        self._last_trade_timestamp = datetime.now(timezone.utc)
        self.logger.info(f"New position set: {position.get('id')}")

    def clear_open_position(self):
        """Clears the current open position."""
        if self._open_position:
            self.logger.info(f"Clearing open position: {self._open_position.get('id')}")
            self._open_position = None
        else:
            self.logger.debug("No open position to clear.")

    def close_position(self, finalized_trade: Dict[str, Any]):
        """Adds a completed trade to history, updates capital, and clears the open position."""
        if finalized_trade:
            self._trade_history.append(finalized_trade)
            net_pnl = finalized_trade.get('net_pnl', 0.0)
            self._current_capital += net_pnl
            self.logger.info(f"Trade {finalized_trade.get('id')} closed. PnL: {net_pnl:.2f}. New Capital: {self._current_capital:.2f}")
        self.clear_open_position()

    def get_last_processed_timestamp(self) -> Optional[datetime]:
        """Gets the timestamp of the last fully processed candle."""
        return self._last_processed_timestamp

    def set_last_processed_timestamp(self, timestamp: Optional[datetime]):
        """Sets the timestamp of the last fully processed candle."""
        self._last_processed_timestamp = timestamp

    def can_open_new_trade(self, current_timestamp: datetime) -> bool:
        """Checks if the bot is allowed to open a new trade based on session rules."""
        if self._open_position:
            self.logger.debug("Cannot open new trade: A position is already open.")
            return False

        if self._last_trade_timestamp:
            cooldown_seconds = self.general_config.min_trade_loop_interval_seconds
            time_since_last_trade = current_timestamp - self._last_trade_timestamp
            if time_since_last_trade < timedelta(seconds=cooldown_seconds):
                self.logger.debug(f"Cannot open new trade: In cooldown period. {time_since_last_trade.total_seconds():.1f}s elapsed.")
                return False

        return True

    def get_state_as_dict(self) -> Dict[str, Any]:
        """Serializes the manager's current state into a dictionary for persistence."""
        return {
            "current_capital": self._current_capital,
            "open_position": self._open_position,
            "trade_history": self._trade_history,
            "last_processed_timestamp": self._last_processed_timestamp.isoformat() if self._last_processed_timestamp else None,
            "last_trade_timestamp": self._last_trade_timestamp.isoformat() if self._last_trade_timestamp else None,
        }

    def load_state_from_dict(self, state: Dict[str, Any]):
        """Loads the manager's state from a deserialized dictionary."""
        self._current_capital = state.get("current_capital", self.trading_config.risk.initial_capital)
        self._open_position = state.get("open_position")
        self._trade_history = state.get("trade_history", [])
        
        last_processed_ts_str = state.get("last_processed_timestamp")
        self._last_processed_timestamp = datetime.fromisoformat(last_processed_ts_str) if last_processed_ts_str else None
        
        last_trade_ts_str = state.get("last_trade_timestamp")
        self._last_trade_timestamp = datetime.fromisoformat(last_trade_ts_str) if last_trade_ts_str else None

        self.logger.info(f"State loaded. Capital: {self._current_capital:.2f}. Open Position: {'Yes' if self._open_position else 'No'}.")