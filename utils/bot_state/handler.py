# utils/bot_state/handler.py

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd # For handling pd.NaT and pd.Timestamp

# Import centralized path configurations
from config.paths import PATHS

logger = logging.getLogger(__name__)

class BotStateHandler:
    """
    Manages the loading and saving of the trading bot's operational state.
    This includes the bot's capital and any active trade positions.
    Ensures state can be persisted and restored reliably.
    """

    def __init__(self,
                 symbol: str,
                 interval: str,
                 model_type: str,
                 initial_capital: float,
                 paths_override: Optional[Dict[str, Union[str, Path]]] = None):
        """
        Initializes the BotStateHandler.

        Args:
            symbol (str): The trading pair symbol.
            interval (str): The data interval.
            model_type (str): The type of model used.
            initial_capital (float): The starting capital for the bot (used if no state is loaded).
            paths_override (Optional[Dict]): Optional overrides for default PATHS config.
        """
        self.symbol = symbol.replace('/', '') # Sanitize symbol for path
        self.interval = interval
        self.model_type = model_type
        self.initial_capital = initial_capital

        self.paths = PATHS.copy()
        if paths_override:
            self.paths.update(paths_override)
        
        # Define the specific state file path
        self.state_file_path = self._get_state_file_path()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"BotStateHandler initialized for {symbol}-{interval}-{model_type}. State file: {self.state_file_path}")

    def _get_state_file_path(self) -> Path:
        """Constructs the full path for the bot's state file."""
        state_dir = self.paths.get('live_trading_state_dir')
        if not state_dir:
            raise ValueError("live_trading_state_dir not configured in paths.py.")
        
        if isinstance(state_dir, str):
            state_dir = Path(state_dir)
        
        state_dir.mkdir(parents=True, exist_ok=True)
        filename = f"bot_state_{self.symbol}_{self.interval}_{self.model_type}.json"
        return state_dir / filename

    def load_state(self) -> Dict[str, Any]:
        """
        Loads the bot's state from a JSON file.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded state.
                            Returns a default initial state if no file is found or loading fails.
        """
        if not self.state_file_path.exists():
            self.logger.info(f"No existing bot state file found at {self.state_file_path}. Initializing with default state.")
            return self._get_default_state()

        try:
            with open(self.state_file_path, 'r') as f:
                state = json.load(f)
            self.logger.info(f"Bot state loaded successfully from {self.state_file_path}.")

            # Convert entry_time back to pandas Timestamp if it was serialized
            if 'current_trade_state' in state and state['current_trade_state']:
                if 'entry_time' in state['current_trade_state'] and state['current_trade_state']['entry_time'] is not None:
                    state['current_trade_state']['entry_time'] = pd.to_datetime(state['current_trade_state']['entry_time'], utc=True)
                else:
                    state['current_trade_state']['entry_time'] = pd.NaT # Ensure it's NaT if None/empty
            
            # Ensure trade_max_holding_bars is Optional[int]
            if 'current_trade_state' in state and state['current_trade_state']:
                if 'trade_max_holding_bars' in state['current_trade_state'] and state['current_trade_state']['trade_max_holding_bars'] is not None:
                    try:
                        state['current_trade_state']['trade_max_holding_bars'] = int(state['current_trade_state']['trade_max_holding_bars'])
                    except (ValueError, TypeError):
                        state['current_trade_state']['trade_max_holding_bars'] = None
                        self.logger.warning("Could not convert 'trade_max_holding_bars' to int. Setting to None.")
                else:
                    state['current_trade_state']['trade_max_holding_bars'] = None

            return state

        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON from state file {self.state_file_path}: {e}. Returning default state.", exc_info=True)
            return self._get_default_state()
        except Exception as e:
            self.logger.error(f"An unexpected error occurred while loading state from {self.state_file_path}: {e}. Returning default state.", exc_info=True)
            return self._get_default_state()

    def save_state(self, current_balance: float, current_trade_state: Dict[str, Any]):
        """
        Saves the bot's current state to a JSON file.

        Args:
            current_balance (float): The current balance of the bot's account.
            current_trade_state (Dict[str, Any]): A dictionary representing the current open trade's state.
        """
        state_to_save = {
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'current_balance': current_balance,
            'current_trade_state': current_trade_state
        }

        # Convert pandas Timestamp to ISO format string for JSON serialization
        if 'entry_time' in state_to_save['current_trade_state'] and pd.notna(state_to_save['current_trade_state']['entry_time']):
            state_to_save['current_trade_state']['entry_time'] = state_to_save['current_trade_state']['entry_time'].isoformat()
        else:
            state_to_save['current_trade_state']['entry_time'] = None # Ensure None if NaT

        try:
            with open(self.state_file_path, 'w') as f:
                json.dump(state_to_save, f, indent=4)
            self.logger.info(f"Bot state saved successfully to {self.state_file_path}.")
        except Exception as e:
            self.logger.error(f"Error saving bot state to {self.state_file_path}: {e}", exc_info=True)

    def _get_default_state(self) -> Dict[str, Any]:
        """Returns the default initial state for the bot."""
        return {
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'current_balance': self.initial_capital,
            'current_trade_state': {
                'position_direction': 0,
                'entry_price': None,
                'entry_time': None,
                'position_asset_qty': 0.0,
                'position_value_entry_usd': 0.0,
                'liquidation_price': None,
                'current_trade_sl_price': None,
                'current_trade_tp_price': None,
                'trade_open_bar_index': -1,
                'trade_max_holding_bars': None,
            }
        }
