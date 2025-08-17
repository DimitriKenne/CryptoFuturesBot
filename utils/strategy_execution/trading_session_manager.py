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
    Designed to be used by both backtesting simulations and live trading bots.
    Provides persistence for live trading sessions.
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
        self.paths = PATHS # Access to global paths configuration

        # Extract relevant config parameters (needed for loading/saving logic if based on config)
        self.initial_capital = float(self.config.trading.risk.initial_capital)
        # Note: Other config rates (leverage, fees, margins) are for TradeExecutionEngine's calculations,
        # not directly for session state management here, but useful for context if needed.


        # Internal state variables
        self._current_capital: float = self.initial_capital
        self._open_position: Optional[Dict[str, Any]] = None # Details of the single open trade
        self._trade_history: List[Dict[str, Any]] = [] # List of completed trade records
        # Equity curve will be a Series, indexed by datetime, with initial capital at start
        self._equity_curve: pd.Series = pd.Series(dtype=float)
        # Store initial equity for cases where equity curve starts before any data points
        self._equity_curve.loc[datetime.now(timezone.utc)] = self.initial_capital


        self.logger.info(f"TradingSessionManager initialized with initial capital: {self.initial_capital:.2f}")

    # ====================================================================
    # --- State Management Methods ---
    # ====================================================================

    def load_state(self, symbol: str, interval: str, model_type: str, is_live_trading: bool):
        """
        Loads the saved session state (capital, open position, and trade history).
        This method is primarily intended for live trading to resume from a previous session.

        Args:
            symbol (str): The trading symbol (e.g., "BTCUSDT").
            interval (str): The data interval (e.g., "1h").
            model_type (str): The type of model used (e.g., "xgboost", "lstm").
            is_live_trading (bool): True if loading for a live trading session, False otherwise.
                                    Affects the paths used.
        """
        if not is_live_trading:
            self.logger.info("Skipping state load: not in live trading mode.")
            return

        self.logger.info("Attempting to load previous trading session state for live trading...")
        state_dir = self.paths.get('live_trading_results_dir')
        capital_state_pattern = self.paths.get('live_trading_capital_state_pattern')
        trades_pattern = self.paths.get('live_trading_trades_pattern')

        if not state_dir or not capital_state_pattern or not trades_pattern:
            self.logger.warning("State/history save paths or patterns not configured in config/paths.py. Starting fresh.")
            return

        try:
            state_dir_path = Path(state_dir)
            # --- Load Internal Capital and Current Position (JSON) ---
            capital_state_filename = capital_state_pattern.format(
                symbol=symbol.replace('/', ''), interval=interval, model_type=model_type
            )
            capital_state_filepath = state_dir_path / capital_state_filename
            if capital_state_filepath.exists():
                with open(capital_state_filepath, 'r') as f:
                    bot_state = json.load(f)
                loaded_capital = bot_state.get('internal_capital')
                if loaded_capital is not None and isinstance(loaded_capital, (int, float)) and loaded_capital > 0:
                    self._current_capital = float(loaded_capital)
                    self.logger.info(f"Loaded previous internal capital: {self._current_capital:.2f}")
                else:
                    self.logger.warning(f"Found state file but 'internal_capital' invalid. Using initial_capital from config.")
                    self._current_capital = self.initial_capital

                loaded_position = bot_state.get('current_position')
                if loaded_position:
                    # Convert entryTime back to datetime if it was stored as string
                    entry_time_str = loaded_position.get('entryTime')
                    if isinstance(entry_time_str, str):
                        try:
                            # Use pd.Timestamp for robust parsing of ISO format strings, then to pydatetime
                            loaded_position['entryTime'] = pd.Timestamp(entry_time_str, tz='UTC').to_pydatetime()
                        except Exception as e:
                            self.logger.error(f"Error parsing entryTime '{entry_time_str}': {e}. Entry time for position will be invalid.", exc_info=True)
                            loaded_position['entryTime'] = None # Set to None if parsing fails
                    self._open_position = loaded_position
                    self.logger.info("Loaded previous open position details.")
                else:
                    self.logger.info("No previous open position found in state file.")
                    self._open_position = None
            else:
                self.logger.info("No previous capital state file found. Starting fresh (initial capital, no open position).")
                self._current_capital = self.initial_capital
                self._open_position = None

            # --- Load Trade History (Parquet) ---
            trades_filename = trades_pattern.format(
                symbol=symbol.replace('/', ''), interval=interval, model_type=model_type
            )
            trades_filepath = state_dir_path / trades_filename
            if trades_filepath.exists():
                try:
                    trades_df = pd.read_parquet(trades_filepath)
                    # Convert datetime columns back to datetime objects for consistency
                    for col in ['entry_time', 'exit_time']:
                        if col in trades_df.columns and pd.api.types.is_datetime64_any_dtype(trades_df[col]):
                            # Ensure UTC timezone if not already
                            if trades_df[col].dt.tz is None:
                                trades_df[col] = trades_df[col].dt.tz_localize('UTC')
                            elif str(trades_df[col].dt.tz) != 'UTC':
                                trades_df[col] = trades_df[col].dt.tz_convert('UTC')
                    self._trade_history = trades_df.to_dict('records')
                    self.logger.info(f"Loaded {len(self._trade_history)} previous trades from: {trades_filepath}")
                except pd.errors.EmptyDataError:
                    self.logger.warning(f"Trade history Parquet file empty: {trades_filepath}.")
                    self._trade_history = []
                except Exception as e:
                    self.logger.error(f"Error loading trade history from {trades_filepath}: {e}", exc_info=True)
                    self._trade_history = []
            else:
                self.logger.info("No previous trade history Parquet file. Starting empty history.")
                self._trade_history = []

            # Initialize equity curve based on loaded history
            if self._trade_history:
                # Get the last recorded capital from the trade history
                last_trade_capital = self._trade_history[-1].get('finalCapital', self._current_capital)
                last_trade_time = self._trade_history[-1].get('exit_time')
                if not isinstance(last_trade_time, datetime): # Fallback if not datetime
                     last_trade_time = datetime.now(timezone.utc)

                # Initialize with initial capital, then update with the last known capital
                self._equity_curve = pd.Series(index=[self._equity_curve.index[0], last_trade_time],
                                               data=[self.initial_capital, last_trade_capital],
                                               dtype=float)
                # Ensure index is unique and sorted
                self._equity_curve = self._equity_curve.loc[~self._equity_curve.index.duplicated(keep='last')].sort_index()

            self.logger.info(f"TradingSessionManager state loaded. Current capital: {self._current_capital:.2f}, Open position: {self._open_position is not None}, Trades in history: {len(self._trade_history)}")

        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding capital state JSON: {e}. Starting fresh.", exc_info=True)
            self._current_capital = self.initial_capital
            self._open_position = None
            self._trade_history = []
        except Exception as e:
            self.logger.error(f"Error loading previous bot state: {e}. Starting fresh.", exc_info=True)
            self._current_capital = self.initial_capital
            self._open_position = None
            self._trade_history = []


    def save_state(self, symbol: str, interval: str, model_type: str, is_live_trading: bool):
        """
        Saves the current session state (capital, open position, and trade history).
        This method is primarily intended for live trading for persistence.

        Args:
            symbol (str): The trading symbol (e.g., "BTCUSDT").
            interval (str): The data interval (e.g., "1h").
            model_type (str): The type of model used (e.g., "xgboost", "lstm").
            is_live_trading (bool): True if saving for a live trading session, False otherwise.
                                    Affects the paths used.
        """
        if not is_live_trading:
            self.logger.debug("Skipping state save: not in live trading mode.")
            return

        self.logger.debug("Saving current trading session state...")
        state_dir = self.paths.get('live_trading_results_dir')
        capital_state_pattern = self.paths.get('live_trading_capital_state_pattern')
        trades_pattern = self.paths.get('live_trading_trades_pattern')

        if not state_dir or not capital_state_pattern or not trades_pattern:
            self.logger.error("State/history save paths/patterns not configured. Cannot save.")
            return

        state_dir_path = Path(state_dir)
        try:
            state_dir_path.mkdir(parents=True, exist_ok=True)

            # --- Save Internal Capital and Current Position (JSON) ---
            current_position_to_save = None
            if self._open_position:
                current_position_to_save = copy.deepcopy(self._open_position)
                # Convert datetime objects to ISO format strings for JSON serialization
                if 'entryTime' in current_position_to_save and isinstance(current_position_to_save['entryTime'], datetime):
                    current_position_to_save['entryTime'] = current_position_to_save['entryTime'].isoformat()
                # Handle numpy types if any
                for key, value in current_position_to_save.items():
                    if isinstance(value, (np.integer, np.floating)):
                        current_position_to_save[key] = value.item() # Convert numpy types to native Python types


            capital_state = {
                'internal_capital': self._current_capital,
                'current_position': current_position_to_save
            }

            capital_state_filename = capital_state_pattern.format(
                symbol=symbol.replace('/', ''), interval=interval, model_type=model_type
            )
            capital_state_filepath = state_dir_path / capital_state_filename
            with open(capital_state_filepath, 'w') as f:
                json.dump(capital_state, f, indent=4)
            self.logger.debug(f"TradingSessionManager capital state and position saved to {capital_state_filepath}")

            # --- Save Trade History (Parquet) ---
            if self._trade_history:
                trades_filename = trades_pattern.format(
                    symbol=symbol.replace('/', ''), interval=interval, model_type=model_type
                )
                trades_filepath = state_dir_path / trades_filename

                new_trades_df = pd.DataFrame(self._trade_history)

                # Ensure datetime columns are timezone-aware (UTC) before saving
                for col in ['entry_time', 'exit_time']:
                    if col in new_trades_df.columns and pd.api.types.is_datetime64_any_dtype(new_trades_df[col]):
                        if new_trades_df[col].dt.tz is None:
                            new_trades_df[col] = new_trades_df[col].dt.tz_localize('UTC')
                        elif str(new_trades_df[col].dt.tz) != 'UTC':
                            new_trades_df[col] = new_trades_df[col].dt.tz_convert('UTC')

                # Handle potential non-string order IDs or None/NaN values for Parquet compatibility
                order_id_columns = ['entryOrderId', 'exitOrderId', 'slOrderId', 'tpOrderId']
                for col in order_id_columns:
                    if col in new_trades_df.columns:
                        new_trades_df[col] = new_trades_df[col].apply(lambda x: str(x) if pd.notna(x) else None)


                existing_trades_df = pd.DataFrame()
                if trades_filepath.exists():
                    try:
                        existing_trades_df = pd.read_parquet(trades_filepath)
                        # Ensure timezone and type consistency for existing DataFrame
                        for col in ['entry_time', 'exit_time']:
                            if col in existing_trades_df.columns and pd.api.types.is_datetime64_any_dtype(existing_trades_df[col]):
                                if existing_trades_df[col].dt.tz is None:
                                    existing_trades_df[col] = existing_trades_df[col].dt.tz_localize('UTC')
                                elif str(existing_trades_df[col].dt.tz) != 'UTC':
                                    existing_trades_df[col] = existing_trades_df[col].dt.tz_convert('UTC')
                        for col in order_id_columns:
                            if col in existing_trades_df.columns:
                                existing_trades_df[col] = existing_trades_df[col].apply(lambda x: str(x) if pd.notna(x) else None)
                    except Exception as e:
                        self.logger.warning(f"Error loading existing trade history from {trades_filepath}: {e}. Will overwrite.", exc_info=True)
                        existing_trades_df = pd.DataFrame() # Start fresh if load fails

                # Concatenate new trades with existing ones, remove duplicates by entry_time for safety
                if not existing_trades_df.empty:
                    combined_trades_df = pd.concat([existing_trades_df, new_trades_df], ignore_index=True)
                    # Drop duplicates based on a combination of critical unique identifiers
                    # For trades, (entry_time, direction, quantity) should be reasonably unique
                    # Or ideally a trade_id if one were generated at entry.
                    # For now, rely on entry_time as primary key for deduplication.
                    combined_trades_df.drop_duplicates(subset=['entry_time'], keep='last', inplace=True)
                else:
                    combined_trades_df = new_trades_df

                if not combined_trades_df.empty:
                    combined_trades_df.to_parquet(trades_filepath, index=False, compression='snappy', engine='pyarrow')
                    self.logger.debug(f"Trade history (total {len(combined_trades_df)} unique trades) saved to {trades_filepath}.")
                    self._trade_history.clear() # Clear in-memory buffer after saving
                else:
                    self.logger.info("No trade history to save to Parquet.")
            else:
                self.logger.debug("No new trades in history buffer to save.")

        except Exception as e:
            self.logger.error(f"Failed to save trading session state: {e}", exc_info=True)


    # ====================================================================
    # --- Position and History Management ---
    # ====================================================================

    def set_open_position(self, trade_details: Dict[str, Any]):
        """
        Sets the details of the currently open position.

        Args:
            trade_details (Dict[str, Any]): A dictionary containing all details of the open trade.
                                            This should typically come from TradeExecutionEngine's
                                            calculate_entry_details output.
        """
        if self._open_position:
            self.logger.warning("Overwriting existing open position. This should ideally not happen without closing the previous one first.")
        self._open_position = trade_details
        self.logger.debug(f"Open position set: {self._open_position.get('direction')} {self._open_position.get('quantity'):.8f} @ {self._open_position.get('entry_price'):.8f}")

    def clear_open_position(self):
        """Clears the details of the currently open position."""
        if self._open_position:
            self.logger.debug(f"Open position cleared: {self._open_position.get('direction')} {self._open_position.get('quantity'):.8f}")
            self._open_position = None
        else:
            self.logger.debug("No open position to clear.")

    def add_completed_trade(self, completed_trade_details: Dict[str, Any]):
        """
        Adds a completed trade record to the session's history and updates capital.

        Args:
            completed_trade_details (Dict[str, Any]): A dictionary containing all finalized
                                                     details of the closed trade, typically
                                                     from TradeExecutionEngine's calculate_exit_details.
        """
        if completed_trade_details:
            self._trade_history.append(completed_trade_details)
            net_pnl = completed_trade_details.get('net_pnl', 0.0)
            if pd.isna(net_pnl):
                net_pnl = 0.0 # Treat NaN PnL as 0 for capital update
                self.logger.warning(f"Net PnL for trade {completed_trade_details.get('exit_reason')} is NaN. Not updating capital for this trade.")

            self._current_capital += net_pnl # Update capital with realized PnL
            self.logger.debug(f"Trade added to history. Reason: {completed_trade_details.get('exit_reason')}. Net PnL: {net_pnl:.2f}. New capital: {self._current_capital:.2f}")
        else:
            self.logger.warning("Attempted to add empty or invalid trade details to history. Capital not updated.")

    def update_equity_curve(self, current_equity: float, timestamp: Union[datetime, pd.Timestamp]):
        """
        Updates the equity curve with the current equity value at a given timestamp.
        This is typically called at the end of each bar processing cycle.

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

        # Use .loc to ensure the index is added/updated correctly, handling DatetimeIndex
        self._equity_curve.loc[timestamp] = current_equity
        # Ensure the equity curve is always sorted by its index
        self._equity_curve = self._equity_curve.sort_index()
        self.logger.debug(f"Equity curve updated at {timestamp}: {current_equity:.2f}")


    # ====================================================================
    # --- Getters for Current State and History ---
    # ====================================================================

    def get_current_capital(self) -> float:
        """Returns the current simulated cash balance."""
        return self._current_capital

    def get_current_equity(self, current_price: Optional[float] = None) -> float:
        """
        Returns the current total equity (capital + unrealized PnL).
        If an open position exists and current_price is provided, unrealized PnL is calculated.
        Otherwise, it returns the last known equity or current capital.
        """
        if self._open_position and current_price is not None and current_price > FLOAT_EPSILON:
            unrealized_pnl = self.calculate_unrealized_pnl(current_price)
            return self._current_capital + unrealized_pnl
        elif not self._equity_curve.empty:
            # Return the latest recorded equity if no current price for open position or no open position
            return self._equity_curve.iloc[-1]
        else:
            return self._current_capital # Fallback to initial capital


    def get_open_position(self) -> Optional[Dict[str, Any]]:
        """Returns the details of the currently open position, or None if flat."""
        return self._open_position

    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Returns the list of all completed trade records."""
        return self._trade_history

    def get_equity_curve(self) -> pd.Series:
        """Returns the complete equity curve as a pandas Series."""
        return self._equity_curve

    # ====================================================================
    # --- Financial Calculation Helpers (re-used for reporting/live PnL) ---
    # ====================================================================

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculates the unrealized Profit/Loss for the current open position.

        Args:
            current_price (float): The current market price.

        Returns:
            float: The calculated unrealized PnL. Returns 0.0 if no open position
                   or if input/position data is invalid.
        """
        if not self._open_position or pd.isna(current_price) or current_price <= FLOAT_EPSILON:
            return 0.0

        entry_price = self._open_position.get('entry_price') # Corrected key
        quantity = self._open_position.get('quantity')
        direction_str = self._open_position.get('direction_str') # 'long' or 'short'

        if not direction_str or pd.isna(entry_price) or entry_price <= FLOAT_EPSILON or pd.isna(quantity) or quantity <= FLOAT_EPSILON:
            self.logger.warning(f"Cannot calculate unrealized PnL: Invalid open position data. EntryPrice: {entry_price}, Qty: {quantity}, Dir: {direction_str}")
            return 0.0

        direction_int = 1 if direction_str == 'long' else -1
        unrealized_pnl = (current_price - entry_price) * quantity * direction_int
        return unrealized_pnl

