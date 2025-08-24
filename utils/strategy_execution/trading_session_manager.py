# utils/strategy_execution/trading_session_manager.py

import pandas as pd
import numpy as np
import json
import copy
import logging
import time # For time.time() in saving
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timezone

# Import configuration from params.py and paths.py
try:
    from config.params import AppConfig, FLOAT_EPSILON
    from config.paths import PATHS
except ImportError as e:
    logging.critical(f"Failed to import necessary configuration modules: {e}. Ensure config/params.py and config/paths.py exist and are correctly structured.", exc_info=True)
    raise # Re-raise to prevent unconfigured manager from being used

logger = logging.getLogger(__name__)

class TradingSessionManager:
    """
    Manages the financial state, open positions, and trade history for a trading session.
    This class is a pure in-memory state manager. It does not perform any file I/O.
    """

    def __init__(self, app_config: AppConfig):
        """
        Initializes the TradingSessionManager.

        Args:
            app_config (AppConfig): The comprehensive application configuration object.
                                    It is assumed that this app_config has already been
                                    validated by config/validator.py externally.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing TradingSessionManager...")

        self.config = app_config

        # Extract relevant config parameters
        self.initial_capital = float(self.config.trading.risk.initial_capital)
        
        # Internal state variables
        self._current_capital: float = self.initial_capital
        self._open_position: Optional[Dict[str, Any]] = None # Details of the single open trade
        self._trade_history: List[Dict[str, Any]] = [] # List of completed trade records
        self._equity_curve: pd.Series = pd.Series(dtype=float) # Initialized empty

        self.logger.info(f"TradingSessionManager initialized with initial capital: {self.initial_capital:.2f}")

    # ====================================================================
    # --- State Management Methods ---
    # ====================================================================
    
    def initialize_equity_curve(self, first_bar_time: Union[datetime, pd.Timestamp]):
        """
        Initializes the equity curve for the trading session with the starting capital.

        Args:
            first_bar_time (Union[datetime, pd.Timestamp]): The timestamp of the first bar in the backtest.
        """
        # Ensure the equity curve is empty before initialization
        self._equity_curve = pd.Series(dtype=float)
        # Set the initial capital at the very first timestamp
        self._equity_curve.loc[first_bar_time] = self.initial_capital

    # ====================================================================
    # --- Position and History Management ---
    # ====================================================================

    def set_open_position(self, trade_details: Dict[str, Any]):
        """
        Sets the details of the currently open position.

        Args:
            trade_details (Dict[str, Any]): A dictionary containing all details of the open trade.
        """
        if self._open_position:
            self.logger.warning("Overwriting existing open position. This should ideally not happen without closing the previous one first.")
        self._open_position = trade_details
        self.logger.debug(f"Open position set: {self._open_position.get('direction_str')} {self._open_position.get('quantity'):.8f} @ {self._open_position.get('entry_price'):.8f}")

    def clear_open_position(self):
        """Clears the details of the currently open position."""
        if self._open_position:
            self.logger.debug(f"Open position cleared: {self._open_position.get('direction_str')} {self._open_position.get('quantity'):.8f}")
            self._open_position = None
        else:
            self.logger.debug("No open position to clear.")

    def add_completed_trade(self, completed_trade_details: Dict[str, Any]):
        """
        Adds a completed trade record to the session's history and updates capital.

        Args:
            completed_trade_details (Dict[str, Any]): A dictionary containing all finalized
                                                     details of the closed trade.
        """
        if completed_trade_details:
            self._trade_history.append(completed_trade_details)
            net_pnl = completed_trade_details.get('net_pnl', 0.0)
            if pd.isna(net_pnl):
                net_pnl = 0.0 # Treat NaN PnL as 0 for capital update
                self.logger.warning(f"Net PnL for trade {completed_trade_details.get('exit_reason')} is NaN. Not updating capital for this trade.")

            self._current_capital += net_pnl # Update capital with realized PnL
            self.logger.info(
                f"Trade recorded: {completed_trade_details.get('direction_str','?').upper()} "
                f"Entry @ {completed_trade_details.get('entry_price',0):.4f}, "
                f"Exit @ {completed_trade_details.get('exit_price',0):.4f}, "
                f"NetPnL: {net_pnl:.2f}, Reason: {completed_trade_details.get('exit_reason','?')}, "
                f"Updated Capital: {self._current_capital:.2f}"
            )
        else:
            self.logger.warning("Attempted to add empty or invalid trade details to history. Capital not updated.")

    def update_equity_curve(self, current_equity: float, timestamp: Union[datetime, pd.Timestamp]):
        """
        Updates the equity curve with the current equity value at a given timestamp.

        Args:
            current_equity (float): The current total equity (capital + unrealized PnL).
            timestamp (Union[datetime, pd.Timestamp]): The timestamp for this equity value.
        """
        if pd.isna(current_equity):
            self.logger.warning(f"Attempted to update equity curve with NaN value at {timestamp}. Skipping.")
            return

        if isinstance(timestamp, datetime):
            if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
        elif isinstance(timestamp, pd.Timestamp):
            if timestamp.tz is None:
                timestamp = timestamp.tz_localize('UTC')
            elif str(timestamp.tz) != 'UTC':
                timestamp = timestamp.tz_convert('UTC')
        else:
            self.logger.error(f"Invalid timestamp type for equity curve update: {type(timestamp)}. Skipping.")
            return

        self._equity_curve.loc[timestamp] = current_equity
        self._equity_curve = self._equity_curve.sort_index()
        self.logger.info(f"Equity curve updated: {timestamp} -> {current_equity:.2f}")


    # ====================================================================
    # --- Getters for Current State and History ---
    # ====================================================================

    def get_current_capital(self) -> float:
        """Returns the current simulated cash balance."""
        return self._current_capital

    def get_current_equity(self, current_price: Optional[float] = None) -> float:
        """
        Returns the current total equity (capital + unrealized PnL).
        """
        if self._open_position and current_price is not None and current_price > FLOAT_EPSILON:
            unrealized_pnl = self.calculate_unrealized_pnl(current_price)
            return self._current_capital + unrealized_pnl
        elif not self._equity_curve.empty:
            return self._equity_curve.iloc[-1]
        else:
            return self._current_capital

    def get_open_position(self) -> Optional[Dict[str, Any]]:
        """Returns the details of the currently open position, or None if flat."""
        return self._open_position

    def get_trade_history_df(self) -> pd.DataFrame:
        """Returns the list of all completed trade records as a DataFrame."""
        return pd.DataFrame(self._trade_history)

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Returns the complete equity curve as a pandas DataFrame with an 'equity' column."""
        return self._equity_curve.to_frame(name='equity')

    # ====================================================================
    # --- Financial Calculation Helpers ---
    # ====================================================================

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculates the unrealized Profit/Loss for the current open position.
        """
        if not self._open_position or pd.isna(current_price) or current_price <= FLOAT_EPSILON:
            return 0.0

        entry_price = self._open_position.get('entry_price')
        quantity = self._open_position.get('quantity')
        direction_str = self._open_position.get('direction_str')

        if not direction_str or pd.isna(entry_price) or entry_price <= FLOAT_EPSILON or pd.isna(quantity) or quantity <= FLOAT_EPSILON:
            self.logger.warning(f"Cannot calculate unrealized PnL: Invalid open position data. EntryPrice: {entry_price}, Qty: {quantity}, Dir: {direction_str}")
            return 0.0

        direction_int = 1 if direction_str == 'long' else -1
        unrealized_pnl = (current_price - entry_price) * quantity * direction_int
        return unrealized_pnl

    def close_position(self, completed_trade_details: Dict[str, Any]):
        """
        Closes the currently open position.
        """
        if not completed_trade_details:
            self.logger.warning("⚠️ Attempted to close position with empty trade details. No action taken.")
            return

        self.add_completed_trade(completed_trade_details)
        self.clear_open_position()

        exit_time = completed_trade_details.get('exit_time')
        final_capital = self.get_current_capital()
        if exit_time is not None:
            self.update_equity_curve(final_capital, exit_time)
            self.logger.info(
                f"\n{'-'*30}\n"
                f"✅ Position Closed\n"
                f"Exit @ {completed_trade_details.get('exit_price', 0):.4f} | "
                f"NetPnL: {completed_trade_details.get('net_pnl', 0.0):.2f} | "
                f"Final Capital: {final_capital:.2f} | "
                f"Exit Time: {exit_time}\n"
                f"{'-'*30}"
            )
        else:
            self.logger.warning("⚠️ No exit_time found in completed trade details. Equity curve not updated for this trade.")