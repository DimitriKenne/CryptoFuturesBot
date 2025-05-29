# trading_bot.py

"""
Trading Bot for Futures Markets using a Ternary Classification Model.

Connects to an exchange, fetches data, generates features, gets model signals,
manages positions and orders based on strategy rules, handles errors, and sends notifications.

Updates:
- Saves trade history in Parquet format, compatible with Backtester and ResultsAnalyser.
- Saves bot state (capital and current open position) to a JSON file.
- Uses key names for trade details consistent with ResultsAnalyser expectations.
- Implements periodic state saving in the main loop.
- Updated to align with changes in DataManager, FeatureEngineer, and ModelTrainer,
  particularly in how expected raw feature columns for the model are determined.
- Added configuration parameter 'exit_on_neutral_signal' to control neutral signal exits.
- Added configuration parameters 'allow_long_trades' and 'allow_short_trades' to filter entry signals.
**FIXED**: Modified _get_signal to pass the correct sequence length of data to ModelTrainer.predict
           when using an LSTM model, resolving the "Not enough data points" warning.
**MODIFIED (NEW)**: Implemented confidence score filtering based on model probabilities.
**MODIFIED**: Added configuration parameters for confidence filtering.
**MODIFIED**: _get_signal now fetches probabilities and passes them to _apply_signal_filters.
**MODIFIED**: _apply_signal_filters now includes confidence threshold check.
**FIXED**: Initialized `filled_exit_details` to `None` in `_close_position` to prevent `NameError`.
**MODIFIED (NEW)**: Integrated volatility regime filter logic for entry filtering and max holding period.
**MODIFIED (NEW)**: Added `_send_notification` as an async wrapper for `NotificationManager`.
**MODIFIED (NEW)**: Added `_async_save_current_state` as an async wrapper for `_save_current_state`.
"""


# Import python-dotenv to load environment variables from a .env file
# Make sure you have it installed: pip install python-dotenv
try:
    from dotenv import load_dotenv, find_dotenv
    # Find and load the .env file at the very beginning
    load_dotenv(find_dotenv())
except ImportError:
    logging.warning("python-dotenv not found. Environment variables must be set manually.")
except Exception as e:
    logging.error(f"Error loading .env file: {e}", exc_info=True)

import asyncio
import logging
import sys
import time
import json # For saving/loading state
import copy # For deepcopying config
import math # Import math for floor in _check_max_holding_period
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import pandas as pd
import numpy as np
import os

# Define FLOAT_EPSILON for robust floating-point comparisons
FLOAT_EPSILON = 1e-9

# --- Project Setup ---
# Add project root to Python path for consistent imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Import Configuration ---
# Attempt to import necessary configuration dictionaries and paths
try:
    from config.params import (
        STRATEGY_CONFIG, MODEL_CONFIG, FEATURE_CONFIG,
        EXCHANGE_CONFIG, LABELING_CONFIG # Include LABELING_CONFIG for fallbacks
    )
    from config.paths import PATHS
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import configuration modules (config/params.py or config/paths.py): "
          f"Ensure config/ is correctly structured. Error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: Unexpected error during configuration import: {e}", file=sys.stderr)
    sys.exit(1)


# --- Import Core Components & Utilities ---
try:
    # Specific adapter (can be made dynamic later if needed)
    from adapters.binance_futures_adapter import BinanceFuturesAdapter
    # Core utilities
    from utils.exchange_interface import ExchangeInterface
    from utils.notification_manager import NotificationManager
    from utils.model_trainer import ModelTrainer
    from utils.features_engineer import FeatureEngineer
    from utils.logger_config import setup_rotating_logging
    from utils.data_manager import DataManager # Import DataManager for Parquet handling
    # Custom exceptions
    from utils.exceptions import OrderExecutionError, ExchangeConnectionError, TemporalSafetyError
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import core components (adapters/, utils/). "
          f"Ensure project structure is correct. Error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: Unexpected error during core component import: {e}", file=sys.stderr)
    sys.exit(1)


# --- Configure Rotating Logging ---
# Setup rotating file logging early. Logs will be saved in the directory specified in PATHS.
log_filename_base = f"trading_bot_{STRATEGY_CONFIG.get('symbol', 'UNKNOWN').replace('/', '')}_{STRATEGY_CONFIG.get('interval', 'UNKNOWN')}_{STRATEGY_CONFIG.get('model_type', 'UNKNOWN')}"
try:
    log_dir = PATHS.get('logs_dir')
    if not log_dir or not isinstance(log_dir, Path):
        raise ValueError("Invalid 'logs_dir' defined in config/paths.py")
    log_dir.mkdir(parents=True, exist_ok=True)
    # Pass the full path to the log file base name
    log_file_path_base = log_dir / log_filename_base
    # Configure rotating logging using the utility function
    setup_rotating_logging(str(log_file_path_base), log_level=logging.INFO)
    logger = logging.getLogger(__name__) # Get logger instance for this module
    logger.info(f"Rotating logging configured. Log files base: {log_file_path_base}")
except Exception as e:
    # Fallback to basic stream logging if rotating setup fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(lineno)d]',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to configure rotating logging: {e}. Using basic stdout logging.", exc_info=True)


# --- Early Notification Manager Initialization ---
# Initialize notifier early to catch critical startup errors.
# Configuration (API keys, etc.) is loaded from STRATEGY_CONFIG['notifier_params'].
notifier_instance = None
try:
    notifier_params = STRATEGY_CONFIG.get('notifier_params', {})
    # Validate Telegram config if enabled
    if 'telegram' in notifier_params and notifier_params['telegram'].get("enabled"):
        tg_config = notifier_params['telegram']
        if not tg_config.get("token") or not tg_config.get("chat_id"):
            logger.warning("Telegram enabled in config, but token or chat_id missing. Disabling Telegram notifications.")
            notifier_params['telegram']['enabled'] = False # Disable if credentials missing

    notifier_instance = NotificationManager(notifier_params)
    logger.info("NotificationManager initialized early for critical errors.")
except Exception as e:
    logger.error(f"Failed to initialize NotificationManager early: {e}. Critical error notifications may not be sent.", exc_info=True)
    # notifier_instance remains None


class TradingBot:
    """
    Automated trading bot for Futures Markets using a Ternary Classification Model.

    Connects to an exchange, fetches data, generates features, gets model signals,
    manages positions and orders based on strategy rules, handles errors, and sends notifications.
    Saves trade history in Parquet format and bot state (capital, position) to JSON.
    """

    def __init__(self, config: Dict[str, Any], notification_manager: Optional[NotificationManager]):
        """
        Initializes the TradingBot instance.

        Args:
            config (Dict[str, Any]): The primary strategy configuration dictionary (STRATEGY_CONFIG).
            notification_manager (Optional[NotificationManager]): An initialized NotificationManager instance.
        """
        self.logger = logging.getLogger(__name__) # Assign logger instance
        self.logger.info("--- Initializing TradingBot ---")

        self.config = config # Store the main strategy config
        self.notification_manager = notification_manager
        self.data_manager = DataManager() # Initialize DataManager

        # --- Load and Validate Core Strategy Parameters ---
        try:
            self._load_and_validate_config()
        except (ValueError, KeyError) as e:
            self.logger.critical(f"Configuration validation failed: {e}", exc_info=True)
            self._critical_shutdown(f"Configuration validation failed: {e}")

        self.logger.info(f"Bot configured for: {self.symbol} @ {self.interval} | Model: {self.model_type}")
        self.logger.info(f"Initial Capital: {self.initial_capital:.2f} | Leverage: {self.leverage}x | Risk/Trade: {self.risk_per_trade_pct}%")
        self.logger.info(f"Volatility Adj: {self.volatility_adjustment_enabled} | Trend Filter: {self.trend_filter_enabled}")
        self.logger.info(f"Exit on Neutral Signal: {'ENABLED' if self.exit_on_neutral_signal else 'DISABLED'}")
        # Log the new parameters
        self.logger.info(f"Allow Long Trades: {'YES' if self.allow_long_trades else 'NO'}")
        self.logger.info(f"Allow Short Trades: {'YES' if self.allow_short_trades else 'NO'}")
        # Log the confidence filter parameters
        self.logger.info(f"Confidence Filter Enabled: {'YES' if self.confidence_filter_enabled else 'NO'}")
        if self.confidence_filter_enabled:
             self.logger.info(f"  Confidence Threshold Long: {self.confidence_threshold_long:.2f} ({self.confidence_threshold_long*100:.0f}%)")
             self.logger.info(f"  Confidence Threshold Short: {self.confidence_threshold_short:.2f} ({self.confidence_threshold_short*100:.0f}%)")
        # Log Volatility Regime Filter parameters (NEW)
        self.logger.info(f"Volatility Regime Filter Enabled: {'YES' if self.volatility_regime_filter_enabled else 'NO'}")
        if self.volatility_regime_filter_enabled:
             self.logger.info(f"  Regime Max Holding (0=Low, 1=Med, 2=High): {self.volatility_regime_max_holding_bars}")
             self.logger.info(f"  Allow Trading in Regime: {self.allow_trading_in_volatility_regime}")


        # --- Initialize Exchange Adapter ---
        self.exchange_adapter: Optional[ExchangeInterface] = None
        try:
            self._initialize_exchange_adapter()
        except (ValueError, ConnectionError, RuntimeError, ExchangeConnectionError) as e:
            self.logger.critical(f"Failed to initialize exchange adapter: {e}", exc_info=True)
            self._critical_shutdown(f"Exchange adapter initialization failed: {e}")

        # --- Initialize Feature Engineer ---
        self.feature_engineer: Optional[FeatureEngineer] = None
        # Determine the expected column name for volatility regime from FEATURE_CONFIG defaults
        # Assuming the FeatureEngineer uses the shortest ATR period for the regime calculation
        fe_atr_periods = FEATURE_CONFIG.get('atr_periods', [14]) # Get default ATR periods from FE config
        atr_period_for_regime = min(fe_atr_periods) if fe_atr_periods else 14 # Use min period or default
        self.volatility_regime_col_name = f'volatility_regime' # Feature Engineer names it simply 'volatility_regime'
        try:
            self._initialize_feature_engineer()
        except (ValueError, KeyError, RuntimeError) as e:
             self.logger.critical(f"Failed to initialize Feature Engineer: {e}", exc_info=True)
             self._critical_shutdown(f"Feature Engineer initialization failed: {e}")


        # --- Load Model ---
        self.model_trainer: Optional[ModelTrainer] = None
        # This list will store the raw feature names the loaded model's preprocessor expects.
        # It's determined after the model is loaded.
        self.expected_raw_feature_columns: List[str] = []
        # Sequence length is crucial for LSTM; initialize from config, will be updated from model metadata
        self.sequence_length: int = int(self.config.get('sequence_length_bars', 1)) # Ensure it's int

        try:
            self._load_model() # Loads model and updates attributes like self.sequence_length
        except (FileNotFoundError, ValueError, RuntimeError, ImportError) as e:
            self.logger.critical(f"Failed to load trading model: {e}", exc_info=True)
            self._critical_shutdown(f"Model loading failed: {e}")

        # --- Calculate Total Lookback Needed ---
        self._calculate_total_lookback()

        # --- Initialize State Variables ---
        self.data_buffer: pd.DataFrame = pd.DataFrame()
        self.last_candle_time: Optional[datetime] = None
        self.current_position: Optional[Dict[str, Any]] = None
        self.open_orders: List[Dict[str, Any]] = []
        self.internal_capital: float = float(self.initial_capital)
        self.trade_history_buffer: List[Dict[str, Any]] = []
        self.is_running: bool = False
        self.stop_event = asyncio.Event()
        self.last_save_time = time.time()
        self.save_interval_sec = 300 # Configurable: save state every 5 minutes

        # --- Load Previous State ---
        self._load_previous_state()

        self.logger.info(f"TradingBot initialization complete. Current internal capital: {self.internal_capital:.2f}")
        if self.current_position:
            self.logger.info(f"Loaded existing position: {self.current_position.get('direction')} {self.current_position.get('quantity'):.8f} @ {self.current_position.get('entryPrice'):.8f}")
        else:
            self.logger.info("No existing position loaded.")
        self.logger.info(f"Loaded {len(self.trade_history_buffer)} trades into history buffer.")

        # --- Determine and Log Expected Raw Feature Columns for the Model ---
        # This logic is now moved here, after _load_model() has populated model_trainer.
        if self.model_trainer:
            if self.model_trainer.features_to_use:
                # If 'features_to_use' was specified during model training and saved in metadata
                self.expected_raw_feature_columns = self.model_trainer.features_to_use
                self.logger.info(f"Model was trained using a specific subset of features: {self.expected_raw_feature_columns}")
            elif self.model_trainer.feature_columns_original:
                # Otherwise, use the original feature columns the preprocessor was fitted on
                self.expected_raw_feature_columns = self.model_trainer.feature_columns_original
                self.logger.info(f"Model's preprocessor was trained on these original features: {self.expected_raw_feature_columns}")
            else:
                # Fallback if metadata is incomplete (should ideally not happen)
                self.expected_raw_feature_columns = []
                self.logger.warning("Could not determine expected raw feature columns from loaded model_trainer "
                                    "(features_to_use or feature_columns_original missing in metadata).")
        else:
            # This case should be caught by critical shutdown if model loading fails
            self.expected_raw_feature_columns = []
            self.logger.error("Model trainer not initialized. Cannot determine expected raw feature columns.")

        # Ensure OHLCV are not in the feature list (they are inputs to FeatureEngineer, not model features)
        if self.expected_raw_feature_columns:
            self.expected_raw_feature_columns = [
                col for col in self.expected_raw_feature_columns
                if col not in ['open', 'high', 'low', 'close', 'volume']
            ]

        if not self.expected_raw_feature_columns:
             self.logger.warning("Expected raw feature columns list for the model is empty. Prediction might fail.")
        else:
             self.logger.info(f"Final list of expected raw feature columns for model prediction: {self.expected_raw_feature_columns}")


    def _load_and_validate_config(self):
        """Loads parameters from self.config and validates them."""
        self.logger.debug("Loading and validating configuration...")

        # --- Extract Core Parameters ---
        self.symbol = self.config.get('symbol')
        self.interval = self.config.get('interval')
        self.model_type = self.config.get('model_type')
        self.initial_capital = float(self.config.get('initial_capital', 0.0))
        self.risk_per_trade_pct = float(self.config.get('risk_per_trade_pct', 0.0))
        self.leverage = int(self.config.get('leverage', 1))
        self.trading_fee_rate = float(self.config.get('trading_fee_rate', 0.0))
        # max_holding_period_bars is now primarily determined by volatility regime, but keep as a default/fallback
        self.max_holding_period_bars = self.config.get('max_holding_period_bars')
        self.slippage_tolerance_pct = float(self.config.get('slippage_tolerance_pct', 0.0))
        self.data_lookback_bars = int(self.config.get('data_lookback_bars', 100))
        self.loop_interval_sec = self.config.get('loop_interval_sec')
        self.exchange_type = self.config.get('exchange_type')
        self.min_liq_distance_pct = float(self.config.get('min_liq_distance_pct', 0.0))
        self.exit_on_neutral_signal = bool(self.config.get('exit_on_neutral_signal', True))
        # Load the new parameters
        self.allow_long_trades = bool(self.config.get('allow_long_trades', True))
        self.allow_short_trades = bool(self.config.get('allow_short_trades', True))

        # --- Extract Confidence Filter Parameters ---
        self.confidence_filter_enabled = bool(self.config.get('confidence_filter_enabled', False))
        # Get percentage thresholds and convert to fractions (0.0-1.0)
        confidence_threshold_long_pct = float(self.config.get('confidence_threshold_long_pct', 0.0))
        confidence_threshold_short_pct = float(self.config.get('confidence_threshold_short_pct', 0.0))
        self.confidence_threshold_long = confidence_threshold_long_pct / 100.0
        self.confidence_threshold_short = confidence_threshold_short_pct / 100.0

        # --- Extract Volatility Regime Filter Parameters (NEW) ---
        self.volatility_regime_filter_enabled = bool(self.config.get('volatility_regime_filter_enabled', False))
        self.volatility_regime_max_holding_bars = self.config.get('volatility_regime_max_holding_bars', {0: None, 1: None, 2: None})
        self.allow_trading_in_volatility_regime = self.config.get('allow_trading_in_volatility_regime', {0: True, 1: True, 2: True})


        # --- Extract Volatility Adjustment Parameters ---
        self.volatility_adjustment_enabled = bool(self.config.get('volatility_adjustment_enabled', False))
        self.volatility_window_bars = int(self.config.get('volatility_window_bars', 14))
        self.fixed_take_profit_pct = float(self.config.get('fixed_take_profit_pct', 0.0))
        self.fixed_stop_loss_pct = float(self.config.get('fixed_stop_loss_pct', 0.0))
        self.alpha_take_profit = float(self.config.get('alpha_take_profit', 1.0))
        self.alpha_stop_loss = float(self.config.get('alpha_stop_loss', 1.0))

        # --- Extract Trend Filter Parameters ---
        self.trend_filter_enabled = bool(self.config.get('trend_filter_enabled', False))
        self.trend_filter_ema_period = int(self.config.get('trend_filter_ema_period', 200))

        # --- Extract Sequence Length ---
        # This is crucial. It should be consistent with the loaded model's sequence length.
        # _load_model will update self.sequence_length based on the loaded model's metadata.
        # Here, we load it from config as an initial value.
        self.sequence_length = int(self.config.get('sequence_length_bars', 1)) # Ensure it's int

        # --- Convert Percentages to Fractions for Internal Use ---
        self.risk_per_trade_fraction = self.risk_per_trade_pct / 100.0
        self.slippage_tolerance_fraction = self.slippage_tolerance_pct / 100.0
        self.min_liq_distance_fraction = self.min_liq_distance_pct / 100.0
        self.fixed_take_profit_fraction = self.fixed_take_profit_pct / 100.0
        self.fixed_stop_loss_fraction = self.fixed_stop_loss_pct / 100.0

        # --- Validation Checks ---
        required_strings = ['symbol', 'interval', 'model_type', 'exchange_type']
        for key in required_strings:
            if not self.config.get(key) or not isinstance(self.config.get(key), str):
                raise ValueError(f"Invalid or missing string configuration for '{key}'.")

        if self.initial_capital <= 0: raise ValueError("'initial_capital' must be positive.")
        if not (0 < self.risk_per_trade_fraction <= 1): raise ValueError("'risk_per_trade_pct' must be between 0 (exclusive) and 100 (inclusive).")
        if self.leverage <= 0: raise ValueError("'leverage' must be positive.")
        if not (0 <= self.trading_fee_rate < 1): raise ValueError("'trading_fee_rate' must be between 0 (inclusive) and 1 (exclusive).")
        # Validate default max_holding_period if it's not None
        if self.max_holding_period_bars is not None and (not isinstance(self.max_holding_period_bars, int) or self.max_holding_period_bars <= 0): raise ValueError("'max_holding_period_bars' must be a positive integer or None.")
        if not (0 <= self.slippage_tolerance_fraction < 1): raise ValueError("'slippage_tolerance_pct' must be between 0 (inclusive) and 100 (exclusive).")
        if self.data_lookback_bars <= 0: raise ValueError("'data_lookback_bars' must be a positive integer.")
        if not (0 <= self.min_liq_distance_fraction <= 1): raise ValueError("'min_liq_distance_pct' must be between 0 and 100 (inclusive).")
        if self.loop_interval_sec is not None and (not isinstance(self.loop_interval_sec, (int, float)) or self.loop_interval_sec <= 0): raise ValueError("'loop_interval_sec' must be a positive number or None.")
        if not isinstance(self.exit_on_neutral_signal, bool): raise ValueError("'exit_on_neutral_signal' must be a boolean.")
        # Validate new parameters
        if not isinstance(self.allow_long_trades, bool): raise ValueError("'allow_long_trades' must be a boolean.")
        if not isinstance(self.allow_short_trades, bool): raise ValueError("'allow_short_trades' must be a boolean.")
        if not self.allow_long_trades and not self.allow_short_trades:
             self.logger.warning("Both 'allow_long_trades' and 'allow_short_trades' are False. Bot will not take any trades.")

        # Validate confidence thresholds (now as fractions)
        if not (0.0 <= self.confidence_threshold_long <= 1.0):
             raise ValueError("'confidence_threshold_long_pct' must result in a fraction between 0.0 and 1.0.")
        if not (0.0 <= self.confidence_threshold_short <= 1.0):
             raise ValueError("'confidence_threshold_short_pct' must result in a fraction between 0.0 and 1.0.")

        # Volatility Regime Filter specific validations (NEW)
        if self.volatility_regime_filter_enabled:
            if not isinstance(self.volatility_regime_max_holding_bars, dict) or set(self.volatility_regime_max_holding_bars.keys()) != {0, 1, 2}:
                raise ValueError("'volatility_regime_max_holding_bars' must be a dictionary with keys 0, 1, 2.")
            if not isinstance(self.allow_trading_in_volatility_regime, dict) or set(self.allow_trading_in_volatility_regime.keys()) != {0, 1, 2}:
                 raise ValueError("'allow_trading_in_volatility_regime' must be a dictionary with keys 0, 1, 2.")
            # Validate values in volatility_regime_max_holding_bars
            for regime, holding_period in self.volatility_regime_max_holding_bars.items():
                 if holding_period is not None and (not isinstance(holding_period, int) or holding_period <= 0):
                      raise ValueError(f"Invalid 'max_holding_period_bars' ({holding_period}) for regime {regime}. Must be positive int or None.")


        if self.volatility_adjustment_enabled:
            if self.volatility_window_bars <= 0: raise ValueError("'volatility_window_bars' must be positive.")
            if self.alpha_take_profit < 0: raise ValueError("'alpha_take_profit' must be non-negative.")
            if self.alpha_stop_loss < 0: raise ValueError("'alpha_stop_loss' must be non-negative.")
        if self.fixed_take_profit_fraction <= 0: raise ValueError("'fixed_take_profit_pct' must be positive.")
        if self.fixed_stop_loss_fraction <= 0: raise ValueError("'fixed_stop_loss_pct' must be positive.")

        if self.trend_filter_enabled and self.trend_filter_ema_period <= 0: raise ValueError("'trend_filter_ema_period' must be positive.")
        if self.sequence_length <= 0: raise ValueError("'sequence_length_bars' must be positive.")

        self.logger.debug("Configuration loaded and validated successfully.")


    def _initialize_exchange_adapter(self):
        """Initializes the appropriate exchange adapter based on config."""
        self.logger.info(f"Initializing exchange adapter: {self.exchange_type}")
        exchange_specific_config = EXCHANGE_CONFIG.get(self.exchange_type)
        if not exchange_specific_config:
            raise ValueError(f"Configuration for exchange type '{self.exchange_type}' not found in EXCHANGE_CONFIG.")

        if self.exchange_type == 'binance_futures':
            api_key = exchange_specific_config.get('api_key')
            api_secret = exchange_specific_config.get('api_secret')
            if not api_key or not api_secret:
                raise ValueError("Binance API Key or Secret is missing in config/params.py or environment variables.")
            self.exchange_adapter = BinanceFuturesAdapter(
                api_key=api_key,
                api_secret=api_secret,
                symbol=self.symbol,
                leverage=self.leverage,
                logger=logging.getLogger('BinanceAdapter'),
                config=exchange_specific_config
            )
            self.logger.info(f"Initialized {self.exchange_type} adapter.")
        else:
            raise ValueError(f"Unsupported exchange type: {self.exchange_type}")


    def _initialize_feature_engineer(self):
        """Initializes the FeatureEngineer with combined configuration."""
        self.logger.info("Initializing FeatureEngineer...")
        feature_engineer_config = copy.deepcopy(FEATURE_CONFIG)
        fe_params_to_override = {
            'volatility_adjustment_enabled': self.volatility_adjustment_enabled,
            'volatility_window_bars': self.volatility_window_bars,
            'fixed_take_profit_pct': self.fixed_take_profit_pct,
            'fixed_stop_loss_pct': self.fixed_stop_loss_pct,
            'alpha_take_profit': self.alpha_take_profit,
            'alpha_stop_loss': self.alpha_stop_loss,
            'trend_filter_enabled': self.trend_filter_enabled,
            'trend_filter_ema_period': self.trend_filter_ema_period,
            # Ensure sequence_length_bars is passed to FeatureEngineer config
            'sequence_length_bars': self.sequence_length,
        }
        feature_engineer_config.update(fe_params_to_override)
        if 'temporal_validation' not in feature_engineer_config:
             feature_engineer_config['temporal_validation'] = {}
        feature_engineer_config['temporal_validation']['enabled'] = False # Disable for live trading

        self.feature_engineer = FeatureEngineer(config=feature_engineer_config)
        self.atr_col_name_volatility = f'atr_{self.feature_engineer.config.get("volatility_window_bars", 14)}'
        self.ema_filter_col_name = f'ema_{self.feature_engineer.config.get("trend_filter_ema_period", 200)}'
        # Volatility regime column name is already determined in __init__
        self.logger.info("FeatureEngineer initialized.")
        self.logger.debug(f"FeatureEngineer using config: {self.feature_engineer.config}")


    def _load_model(self):
        """Loads the trained model pipeline (model + preprocessor) and metadata."""
        self.logger.info(f"Loading model '{self.model_type}' for {self.symbol} {self.interval}...")
        model_specific_config = MODEL_CONFIG.get(self.model_type)
        if not model_specific_config:
            raise ValueError(f"Model configuration for key '{self.model_type}' not found in MODEL_CONFIG.")
        if 'model_type' not in model_specific_config:
            model_specific_config['model_type'] = self.model_type

        model_base_path = PATHS.get("trained_models_dir")
        # Note: ModelTrainer.load constructs the full filename pattern internally
        # based on symbol, interval, and model_key (which is self.model_type here).
        # DataManager is used by ModelTrainer for actual loading.

        if not model_base_path: # No need to check pattern here, ModelTrainer handles it
            raise ValueError("Model save path ('trained_models_dir') not configured in config/paths.py.")

        # Initialize a ModelTrainer instance to handle loading
        # The config passed to ModelTrainer here is primarily for identifying the model type and its params
        # if it were to *build* a model, but for loading, it mainly uses the model_type.
        temp_trainer = ModelTrainer(config=model_specific_config)

        # Load the pipeline; the .load() method updates the trainer instance in place.
        # The 'model_key' argument to load should match the key used during saving (typically self.model_type).
        loaded_trainer = temp_trainer.load(
            symbol=self.symbol,
            interval=self.interval,
            model_key=self.model_type # Assuming model_type is used as the key for saving/loading
        )
        self.model_trainer = loaded_trainer

        # --- Update bot attributes based on loaded model metadata ---
        if self.model_trainer.model_type != self.model_type:
             self.logger.warning(f"Loaded model type '{self.model_trainer.model_type}' differs from STRATEGY_CONFIG '{self.model_type}'. Using loaded type.")
             self.model_type = self.model_trainer.model_type

        # Update sequence_length from the loaded model's metadata
        # ModelTrainer.load sets self.sequence_length from metadata.
        if hasattr(self.model_trainer, 'sequence_length') and self.model_trainer.sequence_length is not None:
             if self.model_trainer.sequence_length != self.sequence_length:
                  self.logger.warning(f"Loaded model sequence length ({self.model_trainer.sequence_length}) differs from STRATEGY_CONFIG ({self.sequence_length}). Using loaded model's length.")
                  self.sequence_length = self.model_trainer.sequence_length
        elif self.model_type == 'lstm': # If LSTM, sequence length is crucial
             self.logger.warning("Loaded LSTM model metadata missing 'sequence_length'. Using value from STRATEGY_CONFIG.")
             # self.sequence_length remains as loaded from STRATEGY_CONFIG
        elif self.model_type != 'lstm' and hasattr(self.model_trainer, 'sequence_length') and self.model_trainer.sequence_length is not None and self.model_trainer.sequence_length > 1:
             # Non-LSTM models should not have sequence_length > 1. Log a warning if metadata is inconsistent.
             self.logger.warning(f"Loaded non-LSTM model '{self.model_type}' has sequence_length > 1 ({self.model_trainer.sequence_length}) in metadata. This is unexpected. Forcing sequence_length to 1.")
             self.sequence_length = 1


        # Logging processed feature columns for information (not directly used by bot for subsetting)
        if self.model_trainer.feature_columns_processed:
            self.logger.info(f"Loaded model's preprocessor output (processed features): {len(self.model_trainer.feature_columns_processed)} columns like {self.model_trainer.feature_columns_processed[:3]}...")
        else:
            self.logger.warning("Processed feature columns list not found in loaded model metadata.")

        self.logger.info(f"Model '{self.model_type}' loaded successfully.")
        self.logger.debug(f"Model Sequence Length (after load): {self.sequence_length}")


    def _calculate_total_lookback(self):
        """Calculates the total number of historical bars needed for initialization."""
        feature_lookback = max(0, self.feature_engineer.required_lookback if self.feature_engineer else 0)
        # Ensure we account for the sequence length for models that need it (like LSTM)
        model_sequence_length = max(0, self.sequence_length) # Use self.sequence_length updated from model

        # The total lookback needed must be at least the maximum of feature lookback
        # and model sequence length to ensure enough data is in the buffer
        # for both feature calculation and sequence creation for the latest bar.
        # Add a small buffer just in case.
        INITIAL_DATA_BUFFER = 50 # Keep a buffer beyond the minimum requirements
        self.total_lookback_needed = max(feature_lookback, model_sequence_length) + INITIAL_DATA_BUFFER

        self.logger.info(f"FeatureEngineer lookback: {feature_lookback}, Model sequence: {model_sequence_length}")
        self.logger.info(f"Total lookback needed for initial data fetch: {self.total_lookback_needed} bars (includes {INITIAL_DATA_BUFFER} buffer).")


    def _load_previous_state(self):
        """Loads internal capital, current position, and trade history from previous runs."""
        self.logger.info("Attempting to load previous bot state...")
        state_dir = PATHS.get('live_trading_results_dir')
        capital_state_pattern = PATHS.get('live_trading_capital_state_pattern')
        trades_pattern = PATHS.get('live_trading_trades_pattern')

        if not state_dir or not capital_state_pattern or not trades_pattern:
             self.logger.warning("State/history save paths or patterns not configured in config/paths.py. Starting fresh.")
             return

        try:
            # --- Load Internal Capital and Current Position (JSON) ---
            capital_state_filename = capital_state_pattern.format(
                symbol=self.symbol.replace('/', ''), interval=self.interval, model_type=self.model_type
            )
            capital_state_filepath = state_dir / capital_state_filename
            if capital_state_filepath.exists():
                with open(capital_state_filepath, 'r') as f:
                    bot_state = json.load(f)
                loaded_capital = bot_state.get('internal_capital')
                if loaded_capital is not None and isinstance(loaded_capital, (int, float)) and loaded_capital > 0:
                    self.internal_capital = float(loaded_capital)
                    self.logger.info(f"Loaded previous internal capital: {self.internal_capital:.2f}")
                else:
                    self.logger.warning(f"Found state file but 'internal_capital' invalid. Using initial_capital from config.")
                    self.internal_capital = float(self.initial_capital)
                self.current_position = bot_state.get('current_position') # Can be None
                if self.current_position: self.logger.info("Loaded previous open position details.")
                else: self.logger.info("No previous open position found in state file.")
            else:
                self.logger.info("No previous capital state file. Using initial_capital and no open position.")
                self.internal_capital = float(self.initial_capital)
                self.current_position = None

            # --- Load Trade History (Parquet using DataManager) ---
            trades_filename = trades_pattern.format(
                symbol=self.symbol.replace('/', ''), interval=self.interval, model_type=self.model_type
            )
            trades_filepath = state_dir / trades_filename
            if trades_filepath.exists():
                # DataManager.load_data is expected to handle Parquet by file extension
                # or internal logic if the path points to a Parquet file.
                # The DataManager provided doesn't have a direct load_data(path) method.
                # It has load_data(symbol, interval, data_type).
                # For now, assuming direct pd.read_parquet is acceptable if DataManager isn't used for this.
                # If `self.data_manager.load_data(trades_filepath)` was intended, it implies
                # `load_data` in DataManager should accept a direct path.
                # The bot already has the full path.
                # For simplicity, and given the existing bot code, direct read is fine here.
                # If `self.data_manager.load_data(trades_filepath)` was intended, it implies
                # `load_data` in DataManager should accept a direct path.
                # The provided DataManager's `load_data` takes symbol, interval, data_type.
                # Let's stick to the bot's original direct Parquet load for now.
                try:
                    trades_df = pd.read_parquet(trades_filepath) # Direct load
                    self.trade_history_buffer = trades_df.to_dict('records')
                    self.logger.info(f"Loaded {len(self.trade_history_buffer)} previous trades from: {trades_filepath}")
                except pd.errors.EmptyDataError:
                     self.logger.warning(f"Trade history Parquet file empty: {trades_filepath}.")
                     self.trade_history_buffer = []
                except Exception as e_read_pq:
                     self.logger.error(f"Error loading trade history from {trades_filepath}: {e_read_pq}", exc_info=True)
                     self.trade_history_buffer = []
            else:
                self.logger.info("No previous trade history Parquet file. Starting empty history.")
                self.trade_history_buffer = []

        except json.JSONDecodeError as e:
             self.logger.error(f"Error decoding capital state JSON: {e}. Starting fresh.", exc_info=True)
             self.internal_capital = float(self.initial_capital); self.current_position = None
        except Exception as e:
            self.logger.error(f"Error loading previous bot state: {e}. Starting fresh.", exc_info=True)
            self.internal_capital = float(self.initial_capital); self.current_position = None; self.trade_history_buffer = []


    def _save_current_state(self): # Note: This is synchronous
        """Saves current capital, position (JSON), and trade history (Parquet)."""
        self.logger.debug("Saving current bot state...")
        state_dir = PATHS.get('live_trading_results_dir')
        capital_state_pattern = PATHS.get('live_trading_capital_state_pattern')
        trades_pattern = PATHS.get('live_trading_trades_pattern')

        if not state_dir or not capital_state_pattern or not trades_pattern:
             self.logger.error("State/history save paths/patterns not configured. Cannot save.")
             return

        if isinstance(state_dir, str): # Ensure state_dir is a Path object
            from pathlib import Path
            state_dir = Path(state_dir)

        try:
            if pd.isna(state_dir): # Check if state_dir is pd.NA (from MockPaths)
                self.logger.warning("MockPaths is being used or state_dir is NA, state saving will be skipped.")
                return

            state_dir.mkdir(parents=True, exist_ok=True)

            current_position_to_save = None
            if self.current_position:
                current_position_to_save = copy.deepcopy(self.current_position)
                if 'entryTime' in current_position_to_save and isinstance(current_position_to_save['entryTime'], datetime):
                    current_position_to_save['entryTime'] = current_position_to_save['entryTime'].isoformat()
                for key, value in current_position_to_save.items():
                    if isinstance(value, pd.Timestamp): # Also handle pandas Timestamps
                        current_position_to_save[key] = value.to_pydatetime().isoformat()
                    elif isinstance(value, (np.integer, np.floating)): # Convert numpy numbers
                        current_position_to_save[key] = value.item()


            capital_state = {'internal_capital': self.internal_capital, 'current_position': current_position_to_save}

            capital_state_filename = capital_state_pattern.format(
                symbol=self.symbol.replace('/', ''), interval=self.interval, model_type=self.model_type
            )
            capital_state_filepath = state_dir / capital_state_filename
            with open(capital_state_filepath, 'w') as f:
                json.dump(capital_state, f, indent=4)
            self.logger.debug(f"Bot capital state and position saved to {capital_state_filepath}")

            if self.trade_history_buffer:
                trades_filename = trades_pattern.format(
                    symbol=self.symbol.replace('/', ''), interval=self.interval, model_type=self.model_type
                )
                trades_filepath = state_dir / trades_filename

                new_trades_df = pd.DataFrame(self.trade_history_buffer)

                order_id_columns = ['entryOrderId', 'exitOrderId', 'slOrderId', 'tpOrderId']

                # Ensure datetime columns are correctly typed for Parquet
                if 'entry_time' in new_trades_df.columns:
                    new_trades_df['entry_time'] = pd.to_datetime(new_trades_df['entry_time'], utc=True, errors='coerce')
                if 'exit_time' in new_trades_df.columns:
                    new_trades_df['exit_time'] = pd.to_datetime(new_trades_df['exit_time'], utc=True, errors='coerce')

                # Ensure order ID columns are string type in new_trades_df
                for col in order_id_columns:
                    if col in new_trades_df.columns:
                        new_trades_df[col] = new_trades_df[col].astype(str).replace('nan', pd.NA).replace('None', pd.NA)


                existing_trades_df = pd.DataFrame()
                if trades_filepath.exists():
                    try:
                        existing_trades_df = pd.read_parquet(trades_filepath)
                        if 'entry_time' in existing_trades_df.columns:
                            existing_trades_df['entry_time'] = pd.to_datetime(existing_trades_df['entry_time'], utc=True, errors='coerce')
                        if 'exit_time' in existing_trades_df.columns:
                            existing_trades_df['exit_time'] = pd.to_datetime(existing_trades_df['exit_time'], utc=True, errors='coerce')
                        # Ensure order ID columns are string type in existing_trades_df upon load
                        for col in order_id_columns:
                            if col in existing_trades_df.columns:
                                existing_trades_df[col] = existing_trades_df[col].astype(str).replace('nan', pd.NA).replace('None', pd.NA)
                    except Exception as e_read_pq:
                        self.logger.error(f"Error loading existing trades from {trades_filepath}: {e_read_pq}", exc_info=True)
                        existing_trades_df = pd.DataFrame() # Start fresh if load fails badly

                if not existing_trades_df.empty and not new_trades_df.empty:
                     # Align columns before concat, ensuring consistent dtypes where possible
                     all_cols = existing_trades_df.columns.union(new_trades_df.columns)
                     existing_trades_df = existing_trades_df.reindex(columns=all_cols)
                     new_trades_df = new_trades_df.reindex(columns=all_cols)

                     # Convert to common dtype (object for strings, datetime64[ns, UTC] for times) before concat
                     for col in all_cols:
                         if col in order_id_columns:
                             existing_trades_df[col] = existing_trades_df[col].astype(str).replace('nan', pd.NA).replace('None', pd.NA)
                             new_trades_df[col] = new_trades_df[col].astype(str).replace('nan', pd.NA).replace('None', pd.NA)
                         elif col in ['entry_time', 'exit_time']:
                             existing_trades_df[col] = pd.to_datetime(existing_trades_df[col], utc=True, errors='coerce')
                             new_trades_df[col] = pd.to_datetime(new_trades_df[col], utc=True, errors='coerce')

                     combined_trades_df = pd.concat([existing_trades_df, new_trades_df], ignore_index=True)
                elif not new_trades_df.empty:
                    combined_trades_df = new_trades_df
                else:
                    combined_trades_df = existing_trades_df

                if not combined_trades_df.empty:
                    if 'entry_time' in combined_trades_df.columns: # Critical for valid trades
                        combined_trades_df.dropna(subset=['entry_time'], inplace=True)

                    if not combined_trades_df.empty:
                        # Final explicit cast for order IDs to string, handling pd.NA for missing values
                        for col in order_id_columns:
                            if col in combined_trades_df.columns:
                                # Convert to object first to handle pd.NA correctly with astype(str)
                                combined_trades_df[col] = combined_trades_df[col].astype('object').where(combined_trades_df[col].notna(), None).astype(str)
                                # Pyarrow prefers None for null strings rather than 'nan' or 'None' strings.
                                # Replace string 'None' and 'nan' with actual None for pyarrow if they slipped through.
                                combined_trades_df[col] = combined_trades_df[col].replace({'None': None, 'nan': None})


                        combined_trades_df.to_parquet(trades_filepath, index=False, compression='snappy', engine='pyarrow')
                        self.logger.debug(f"Trade history (total {len(combined_trades_df)}) saved to {trades_filepath}.")
                        self.trade_history_buffer = []
                    else:
                        self.logger.info("Trade history buffer was cleared or became empty after NaT removal; not saving Parquet.")
                else:
                    self.logger.info("No trade data to save to Parquet.")
            # self.last_save_time = time.time() # Assuming time module is imported and last_save_time is an attribute
        except Exception as e:
            self.logger.error(f"Failed to save bot state: {e}", exc_info=True)


    def _critical_shutdown(self, message: str):
        """Handles immediate shutdown logging for critical initialization errors."""
        self.logger.critical(f"CRITICAL SHUTDOWN: {message}")
        if self.notification_manager:
             self.logger.critical(f"CRITICAL STARTUP FAILURE NOTIFICATION (sync log): {message}")
        sys.exit(1)

    # --- Part 2: Bot Operation ---

    async def run(self):
        """Starts the main asynchronous trading bot execution loop."""
        if not self.exchange_adapter or not self.feature_engineer or not self.model_trainer:
             self.logger.critical("Bot components not initialized correctly. Cannot run.")
             self._critical_shutdown("Bot components not initialized.")
             return

        self.is_running = True
        self.logger.info("--- Trading Bot Started ---")
        await self._send_notification("Trading bot started.", level="info")
        await self._send_notification(f"Current internal capital: {self.internal_capital:.2f}", level="info")
        if self.current_position:
             await self._send_notification(f"Loaded existing position: {self.current_position.get('direction')} {self.current_position.get('quantity'):.8f} @ {self.current_position.get('entryPrice'):.8f}", level="info")

        try:
            # Perform async setup for the adapter (e.g., set leverage, fetch initial info)
            await self.exchange_adapter.async_setup()

            # --- Initial Data Fetch ---
            await self._initialize_data_buffer()

            # --- Main Loop ---
            while self.is_running and not self.stop_event.is_set():
                next_candle_time = self._get_next_candle_time()
                if not next_candle_time:
                    self.logger.error("Could not determine next candle time. Pausing and retrying.")
                    await asyncio.sleep(self.loop_interval_sec or 60) # Wait before retrying
                    continue

                # Wait until the next candle should be closed
                await self._wait_until_next_candle(next_candle_time)

                # Fetch latest data and update buffer
                if not await self._update_buffer_with_recent_data():
                    self.logger.warning("Failed to update data buffer. Skipping cycle.")
                    continue # Skip processing if buffer update fails

                # Process the latest candle data (features -> signal -> trade logic)
                await self._process_latest_candle()

                # --- Periodic State Saving ---
                if time.time() - self.last_save_time >= self.save_interval_sec:
                    self.logger.info("Periodic state save triggered.")
                    # Use asyncio.create_task for fire-and-forget save
                    asyncio.create_task(self._async_save_current_state()) # Use async wrapper


                # Small sleep to prevent tight looping if errors occur or interval is very short
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            self.logger.info("Bot run loop cancelled (likely shutdown signal).")
        except Exception as e:
            self.logger.critical(f"Unhandled error in main bot loop: {e}", exc_info=True)
            await self._send_notification(f"CRITICAL: Unhandled error in main loop: {e}", level="critical")
        finally:
            self.logger.info("Main loop exited. Initiating final shutdown sequence.")
            await self.shutdown()


    async def _initialize_data_buffer(self):
        """Fetches initial historical data to populate the buffer."""
        self.logger.info(f"Fetching initial data buffer ({self.total_lookback_needed} candles)...")
        try:
            initial_data = await self.exchange_adapter.fetch_recent_candles(
                symbol=self.symbol,
                interval=self.interval,
                limit=self.total_lookback_needed
            )

            if initial_data is None or initial_data.empty:
                raise ValueError("Failed to fetch initial data buffer (received None or empty).")

            if len(initial_data) < self.total_lookback_needed:
                 self.logger.critical(f"Fetched only {len(initial_data)} candles, less than required ({self.total_lookback_needed}). Buffer may be insufficient initially.")
                 await self._send_notification(f"CRITICAL: Insufficient initial data ({len(initial_data)}/{self.total_lookback_needed}).", level="critical")

            self.data_buffer = initial_data.copy()
            self.last_candle_time = self.data_buffer.index[-1]
            self.logger.info(f"Initial data buffer populated. Size: {len(self.data_buffer)}. Last candle: {self.last_candle_time}")

        except Exception as e:
            self.logger.critical(f"Failed to initialize data buffer: {e}", exc_info=True)
            self._critical_shutdown(f"Failed to initialize data buffer: {e}")


    async def _update_buffer_with_recent_data(self):
        """Fetches the most recent candles and updates the data buffer."""
        self.logger.debug("Fetching recent data to update buffer...")
        try:
            fetch_limit = 10 # Fetch more than 1 to handle potential missed candles
            latest_data = await self.exchange_adapter.fetch_recent_candles(
                symbol=self.symbol,
                interval=self.interval,
                limit=fetch_limit
            )

            if latest_data is None or latest_data.empty:
                self.logger.warning("No recent data fetched to update buffer.")
                return False

            if not isinstance(latest_data.index, pd.DatetimeIndex):
                self.logger.error("Latest data index is not DatetimeIndex. Cannot update buffer.")
                return False

            combined_buffer = pd.concat([self.data_buffer, latest_data], axis=0, sort=False)
            combined_buffer = combined_buffer[~combined_buffer.index.duplicated(keep='last')]
            combined_buffer = combined_buffer.sort_index()
            # Keep only the required number of bars for the buffer
            self.data_buffer = combined_buffer.tail(self.total_lookback_needed).copy()

            new_last_candle_time = self.data_buffer.index[-1]
            if self.last_candle_time is None or new_last_candle_time > self.last_candle_time:
                 self.last_candle_time = new_last_candle_time
                 self.logger.debug(f"Data buffer updated. Size: {len(self.data_buffer)}. New last candle: {self.last_candle_time}")
                 return True
            else:
                 self.logger.debug("No new candle data found in fetched data.")
                 return True

        except Exception as e:
            self.logger.error(f"Error updating data buffer: {e}", exc_info=True)
            return False


    def _get_next_candle_time(self) -> Optional[datetime]:
        """Calculates the expected UTC close time of the next candle."""
        if self.last_candle_time is None:
            self.logger.error("Cannot determine next candle time: last_candle_time is not set.")
            return None
        if not isinstance(self.last_candle_time, pd.Timestamp):
             self.logger.error(f"Invalid last_candle_time type: {type(self.last_candle_time)}")
             return None

        try:
            interval_timedelta = self._parse_interval_to_timedelta(self.interval)
            if interval_timedelta is None:
                raise ValueError(f"Could not parse interval string: {self.interval}")

            last_time_utc = self.last_candle_time.tz_convert(timezone.utc) if self.last_candle_time.tz else self.last_candle_time.tz_localize(timezone.utc)
            next_candle_time_utc = last_time_utc + interval_timedelta
            return next_candle_time_utc

        except Exception as e:
            self.logger.error(f"Error calculating next candle time: {e}", exc_info=True)
            return None


    def _parse_interval_to_timedelta(self, interval_str: str) -> Optional[timedelta]:
        """Parses a Binance interval string (e.g., '1m', '1h') into a timedelta."""
        try:
            if not isinstance(interval_str, str) or len(interval_str) < 2: return None
            unit = interval_str[-1].lower()
            value = int(interval_str[:-1])
            if unit == 'm': return timedelta(minutes=value)
            if unit == 'h': return timedelta(hours=value)
            if unit == 'd': return timedelta(days=value)
            if unit == 'w': return timedelta(weeks=value)
            if unit == 's': return timedelta(seconds=value)
            self.logger.warning(f"Unsupported interval unit '{unit}' for precise timedelta calculation.")
            return None
        except (ValueError, TypeError):
            self.logger.error(f"Failed to parse interval string: {interval_str}")
            return None


    async def _wait_until_next_candle(self, next_candle_time_utc: datetime):
        """Waits until the specified UTC time plus a small buffer."""
        WAIT_BUFFER_SEC = 10
        wait_until_utc = next_candle_time_utc + timedelta(seconds=WAIT_BUFFER_SEC)
        now_utc = datetime.now(timezone.utc)
        time_to_wait_sec = (wait_until_utc - now_utc).total_seconds()

        if time_to_wait_sec > 0:
            self.logger.info(f"Waiting {time_to_wait_sec:.2f}s for next candle close ({next_candle_time_utc.isoformat()})...")
            await asyncio.sleep(time_to_wait_sec)
            self.logger.debug("Wait complete.")
        else:
            lag_seconds = abs(time_to_wait_sec)
            self.logger.warning(f"Next candle time ({next_candle_time_utc.isoformat()}) is in the past. Lagging by {lag_seconds:.2f}s. Processing immediately.")


    async def _process_latest_candle(self):
        """Processes the latest candle: features -> signal -> trade execution."""
        self.logger.info(f"--- Processing Candle: {self.last_candle_time} ---")

        if self.data_buffer.empty:
            self.logger.warning("Data buffer empty, cannot process candle.")
            return

        # --- 1. Feature Engineering ---
        try:
            # Feature engineering requires a certain lookback.
            # The output `featured_data` will have features calculated for all rows in the buffer.
            min_required_for_features = self.feature_engineer.required_lookback + 1 # Add 1 because lookback is N-1 bars
            if len(self.data_buffer) < min_required_for_features:
                 self.logger.warning(f"Insufficient data ({len(self.data_buffer)}/{min_required_for_features}) for feature engineering. Skipping cycle.")
                 return

            # Generate features for the entire buffer; latest features are in the last row
            featured_data = self.feature_engineer.process(self.data_buffer)
            self.logger.debug(f"FeatureEngineer produced {len(featured_data.columns)} columns.")

            # Ensure the volatility regime column is present in the latest features if filter is enabled
            if self.volatility_regime_filter_enabled and self.volatility_regime_col_name not in featured_data.columns:
                 self.logger.error(f"Volatility regime filter enabled, but column '{self.volatility_regime_col_name}' is missing from featured data.")
                 # This is a critical issue for the filter, log but don't necessarily stop the bot
                 # The filter check in _apply_entry_filters will handle the missing column gracefully
                 pass


        except TemporalSafetyError as e:
            self.logger.critical(f"Temporal safety violation during feature engineering: {e}. Stopping bot.", exc_info=True)
            await self._send_notification(f"CRITICAL: Temporal safety violation: {e}. Bot stopping.", level="critical")
            self.is_running = False
            return
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}", exc_info=True)
            await self._send_notification(f"ERROR: Feature engineering failed: {e}", level="error")
            return

        # --- 2. Signal Generation ---
        try:
            # Pass the featured_data (including OHLCV and all generated features)
            # The _get_signal method will handle selecting the correct data slice for the model.
            signal, probabilities = await self._get_signal(featured_data) # MODIFIED: Get probabilities too
            self.logger.info(f"Generated Signal: {signal}")
            if probabilities is not None:
                 self.logger.info(f"Model Probabilities: {probabilities.iloc[-1].to_dict()}") # Log probabilities for the latest bar
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}", exc_info=True)
            await self._send_notification(f"ERROR: Signal generation failed: {e}", level="error")
            return

        # --- 3. Trade Execution ---
        try:
            # Pass the latest features and probabilities to the trade logic
            await self._execute_trade_logic(signal, featured_data.iloc[[-1]], probabilities.iloc[[-1]] if probabilities is not None else None) # MODIFIED: Pass latest features and probabilities
        except Exception as e:
            self.logger.error(f"Trade execution logic failed: {e}", exc_info=True)
            await self._send_notification(f"ERROR: Trade execution failed: {e}", level="error")

        self.logger.info(f"--- Finished Processing Candle: {self.last_candle_time} ---")


    async def _get_signal(self, X_input: pd.DataFrame) -> Tuple[int, Optional[pd.DataFrame]]:
        """
        Generates trading signal and probabilities from features using the loaded model and applies filters.
        Selects the raw features expected by the model before prediction.
        Handles sequence slicing for LSTM models.
        """
        if self.model_trainer is None or not hasattr(self.model_trainer, 'predict') or not hasattr(self.model_trainer, 'predict_proba'): # Check for predict_proba
            self.logger.error("Model trainer or predict/predict_proba method not available.")
            return 0, None # Neutral signal and no probabilities

        if X_input.empty:
            self.logger.warning("Input data for signal generation is empty.")
            return 0, None

        # --- Prepare Data for Prediction based on Model Type ---
        if self.model_type == 'lstm':
            # For LSTM, we need the last 'sequence_length' bars from the buffer.
            # Ensure there are enough data points in the buffer for the sequence.
            if len(X_input) < self.sequence_length:
                self.logger.warning(f"Not enough data points ({len(X_input)}) in buffer to create sequence of length {self.sequence_length}. Cannot predict.")
                return 0, None

            # Select the last 'sequence_length' rows from the featured data buffer
            X_predict_data = X_input.tail(self.sequence_length).copy()
            self.logger.debug(f"Prepared {len(X_predict_data)} data points for LSTM sequence prediction.")

        else:
            # For non-sequence models (like RandomForest, XGBoost), use only the last bar.
            X_predict_data = X_input.iloc[[-1]].copy() # Ensure it's a DataFrame (single row)
            self.logger.debug("Prepared single data point for non-sequence model prediction.")


        # --- Select only the raw feature columns expected by the model's preprocessor ---
        # self.expected_raw_feature_columns was determined during __init__
        if not self.expected_raw_feature_columns:
             self.logger.error("Expected raw feature columns list is empty. Cannot make prediction.")
             return 0, None

        # Check if the data prepared for prediction contains these expected raw columns
        missing_raw_cols = [col for col in self.expected_raw_feature_columns if col not in X_predict_data.columns]
        if missing_raw_cols:
             self.logger.error(f"Prediction input data is missing expected raw feature columns: {missing_raw_cols}. Cannot predict.")
             self.logger.debug(f"Available columns in prediction data: {X_predict_data.columns.tolist()}")
             return 0, None

        # Select only the expected raw feature columns in the correct order
        X_predict_raw_subset = X_predict_data[self.expected_raw_feature_columns].copy()

        self.logger.debug(f"Passing {len(X_predict_raw_subset.columns)} raw columns to model prediction (before pipeline transform).")
        self.logger.debug(f"Columns: {X_predict_raw_subset.columns.tolist()}")


        # --- Data Quality Check (NaN/Inf) on selected raw features ---
        # Check for NaNs/Infs in the data slice *before* passing to the model trainer's predict method.
        # The model trainer's predict method *should* handle this internally, but an explicit check here
        # provides clearer logging specifically for the bot's prediction input.
        if X_predict_raw_subset.isnull().values.any():
             nan_cols = X_predict_raw_subset.columns[X_predict_raw_subset.isnull().any()].tolist()
             self.logger.warning(f"Prediction input data (raw subset) contains NaNs in columns: {nan_cols}. Cannot predict.")
             return 0, None
        numeric_cols = X_predict_raw_subset.select_dtypes(include=np.number).columns
        if not numeric_cols.empty and np.isinf(X_predict_raw_subset[numeric_cols].astype(np.float64).values).any():
             inf_cols = numeric_cols[np.isinf(X_predict_raw_subset[numeric_cols].astype(np.float64).values).any(axis=0)]
             self.logger.warning(f"Prediction input data (raw subset) contains Infs in columns: {inf_cols}. Cannot predict.")
             return 0, None


        # --- Generate Raw Model Prediction and Probabilities ---
        try:
            # Pass the prepared data (either sequence or single row) to the model trainer's predict method.
            # ModelTrainer.predict handles the preprocessing pipeline internally,
            # including reshaping for LSTM if necessary.
            prediction_series = self.model_trainer.predict(X_predict_raw_subset)
            # Get probabilities as well
            probability_df = self.model_trainer.predict_proba(X_predict_raw_subset) # Should return DataFrame with columns -1, 0, 1

            if prediction_series is None or prediction_series.empty:
                 self.logger.warning("Model prediction returned None or empty.")
                 return 0, probability_df # Return probabilities even if prediction is bad

            # Get the signal from the last prediction (for the latest bar)
            raw_signal_val = prediction_series.iloc[-1]
            if pd.isna(raw_signal_val):
                 self.logger.warning("Model prediction is NaN/NA.")
                 return 0, probability_df # Return probabilities even if prediction is bad

            signal = int(raw_signal_val)
            if signal not in [-1, 0, 1]:
                 self.logger.warning(f"Model prediction out of range ({signal}). Clamping to 0.")
                 signal = 0
            self.logger.debug(f"Raw model signal: {signal}")

        except Exception as e:
            self.logger.error(f"Error during model prediction or probability generation: {e}", exc_info=True)
            return 0, None # Return neutral signal and no probabilities on error

        # --- Apply Strategy Filters ---
        # Pass the raw signal, the latest features (for trend filter and volatility regime), and the probabilities
        if signal != 0:
             # Pass the single-row DataFrame for the latest bar and the probability DataFrame
             signal = self._apply_entry_filters(signal, X_input.iloc[[-1]], probability_df.iloc[[-1]] if probability_df is not None else None) # MODIFIED: Pass latest features and probabilities

        return signal, probability_df


    async def _execute_trade_logic(self, signal: int, latest_features: pd.DataFrame, latest_probabilities: Optional[pd.DataFrame]):
        """Handles trade execution based on signal, current position state, and filters."""
        self.logger.debug(f"Executing trade logic for signal: {signal}")

        await self._refresh_position_and_orders()

        # Check for and cancel orphaned orders
        if self.current_position is None and self.open_orders:
            self.logger.warning("Detected open orders but no open position. Cancelling orphaned orders.")
            await self._send_notification("WARNING: Detected orphaned orders. Cancelling...", level="warning")
            await self._cancel_open_orders()
            await self._refresh_position_and_orders()

        position_direction = self.current_position.get('direction') if self.current_position else None
        position_quantity = self.current_position.get('quantity') if self.current_position else 0.0

        # --- Scenario 1: No Position ---
        if position_direction is None:
            # Apply entry filters here, *before* attempting to open a position
            # Pass the latest features and probabilities to the filters
            filtered_signal = self._apply_entry_filters(signal, latest_features, latest_probabilities) # MODIFIED: Apply entry filters

            if filtered_signal == 1:
                # Add check for allowed long trades
                if self.allow_long_trades:
                    self.logger.info("Signal: LONG, Filters Passed. No position open. Attempting to open LONG.")
                    await self._open_position(side='buy')
                else:
                    self.logger.debug("Signal: LONG, Filters Passed. Long trades are DISABLED. Skipping entry.")
            elif filtered_signal == -1:
                # Add check for allowed short trades
                if self.allow_short_trades:
                    self.logger.info("Signal: SHORT, Filters Passed. No position open. Attempting to open SHORT.")
                    await self._open_position(side='sell')
                else:
                    self.logger.debug("Signal: SHORT, Filters Passed. Short trades are DISABLED. Skipping entry.")
            else: # filtered_signal is 0 (either raw signal was 0 or filters blocked it)
                self.logger.debug(f"Signal: {signal}, Filters Blocked. No position open. No action.")


        # --- Scenario 2: Long Position Open ---
        elif position_direction == 'long':
            if signal == 1:
                self.logger.debug("Signal: LONG. Already LONG. Holding position.")
            elif signal == -1:
                # Add check for allowed short trades (for reversal)
                if self.allow_short_trades:
                    self.logger.info("Signal: SHORT. Currently LONG. Attempting to REVERSE (close LONG, open SHORT).")
                    await self._close_position(side='sell', quantity=position_quantity, reason='signal_reverse')
                    await asyncio.sleep(1.0) # Small delay to allow close order to process
                    # After closing, we are flat. The next cycle's _execute_trade_logic will process
                    # the new signal (-1) and potentially open the short position, applying filters.
                    # No need to call _open_position directly here.
                else:
                    self.logger.debug("Signal: SHORT. Currently LONG. Short trades are DISABLED, cannot reverse. Holding LONG.")
            # Modified: Only close on neutral signal if exit_on_neutral_signal is True
            elif signal == 0:
                if self.exit_on_neutral_signal:
                    self.logger.info("Signal: NEUTRAL. Currently LONG. Exit on neutral is ENABLED. Attempting to CLOSE LONG.")
                    await self._close_position(side='sell', quantity=position_quantity, reason='signal_exit_neutral') # Use specific reason
                else:
                    self.logger.debug("Signal: NEUTRAL. Currently LONG. Exit on neutral is DISABLED. Holding position.")


        # --- Scenario 3: Short Position Open ---
        elif position_direction == 'short':
            if signal == 1:
                # Add check for allowed long trades (for reversal)
                if self.allow_long_trades:
                    self.logger.info("Signal: LONG. Currently SHORT. Attempting to REVERSE (close SHORT, open LONG).")
                    await self._close_position(side='buy', quantity=position_quantity, reason='signal_reverse')
                    await asyncio.sleep(1.0) # Small delay to allow close order to process
                     # After closing, we are flat. The next cycle's _execute_trade_logic will process
                    # the new signal (1) and potentially open the long position, applying filters.
                    # No need to call _open_position directly here.
                else:
                    self.logger.debug("Signal: LONG. Currently SHORT. Long trades are DISABLED, cannot reverse. Holding SHORT.")
            elif signal == -1:
                self.logger.debug("Signal: SHORT. Already SHORT. Holding position.")
            # Modified: Only close on neutral signal if exit_on_neutral_signal is True
            elif signal == 0:
                 if self.exit_on_neutral_signal:
                    self.logger.info("Signal: NEUTRAL. Currently SHORT. Exit on neutral is ENABLED. Attempting to CLOSE SHORT.")
                    await self._close_position(side='buy', quantity=position_quantity, reason='signal_exit_neutral') # Use specific reason
                 else:
                    self.logger.debug("Signal: NEUTRAL. Currently SHORT. Exit on neutral is DISABLED. Holding position.")


        # --- Check Max Holding Period ---
        # This check happens regardless of signal, but only if a position is open
        if self.current_position: # Check if a position is open
            # The max holding period is now dynamic based on volatility regime
            await self._check_max_holding_period(latest_features) # MODIFIED: Pass latest features for regime check


    def _apply_entry_filters(self, signal: int, latest_features: pd.DataFrame, latest_probabilities: Optional[pd.DataFrame]) -> int:
        """
        Applies configured filters (e.g., EMA trend, confidence threshold, volatility regime) to an entry signal.
        This is called when the bot is FLAT and receives a non-neutral signal.
        Filters are based on data from the latest bar.
        """
        # Signal is guaranteed to be non-zero here (1 or -1)
        if signal == 0: return 0 # Should not be called with signal 0, but defensive
        if latest_features.empty:
             self.logger.warning("Input data for entry filters is empty.")
             return 0

        current_bar = latest_features.iloc[-1] # Get the single row for the latest bar
        current_timestamp = latest_features.index[-1]
        current_close = current_bar['close']

        # --- Volatility Regime Filter (NEW) ---
        if self.volatility_regime_filter_enabled:
             try:
                  # Get the volatility regime for the current bar
                  # Use .get() with default pd.NA for robustness if column is missing
                  current_regime = current_bar.get(self.volatility_regime_col_name, pd.NA)

                  # Check if the regime value is valid (0, 1, or 2) and not NaN
                  if pd.isna(current_regime) or current_regime not in [0, 1, 2]:
                       self.logger.warning(f"Candle {current_timestamp}: Volatility regime value ({current_regime}) is invalid or NaN. Volatility regime filter blocks entry.")
                       return 0 # Block entry if regime is invalid

                  # Check if trading is allowed in this regime
                  # Use .get() with default False for robustness if regime key is missing (shouldn't happen with validation)
                  if not self.allow_trading_in_volatility_regime.get(current_regime, False):
                       self.logger.debug(f"Candle {current_timestamp}: Trading is not allowed in volatility regime {current_regime}. Volatility regime filter blocks signal {signal}.")
                       return 0 # Block entry if trading is not allowed in this regime
                  self.logger.debug(f"Candle {current_timestamp}: Trading is allowed in volatility regime {current_regime}. Passed volatility regime filter.")

             except KeyError:
                  # This should ideally be caught during feature engineering, but defensive here
                  self.logger.error(f"Volatility regime filter enabled, but column '{self.volatility_regime_col_name}' not found in latest features. Disabling filter and blocking trade.")
                  self.volatility_regime_filter_enabled = False # Disable filter permanently if column missing
                  return 0 # Block trade
             except Exception as e:
                  self.logger.error(f"Error during volatility regime filter check at candle {current_timestamp}: {e}. Blocking entry.", exc_info=True)
                  return 0 # Block entry on filter error


        # --- Confidence Threshold Filter ---
        if self.confidence_filter_enabled:
             # Determine the correct threshold based on the signal direction
             confidence_threshold = self.confidence_threshold_long if signal == 1 else self.confidence_threshold_short

             # If the threshold is 0 or less, the filter is effectively disabled for this direction
             if confidence_threshold > FLOAT_EPSILON:
                  if latest_probabilities is None or latest_probabilities.empty:
                       self.logger.warning("Confidence filter enabled but probabilities are missing or empty. Blocking entry.")
                       return 0 # Block entry if probabilities are missing

                  try:
                       # Get the probability for the predicted signal from the latest_probabilities DataFrame
                       # Use .get() with default 0.0 for robustness if a column is missing (shouldn't happen if ModelTrainer.predict_proba is correct)
                       confidence = latest_probabilities.iloc[-1].get(signal, 0.0)
                       if pd.isna(confidence):
                            self.logger.debug(f"Confidence score is NaN for signal {signal}. Confidence filter blocks entry.")
                            return 0 # Block entry if confidence is NaN
                       if confidence < confidence_threshold:
                            self.logger.debug(f"Confidence {confidence:.2f} for signal {signal} is below threshold {confidence_threshold:.2f}. Confidence filter blocks entry.")
                            return 0 # Block entry if confidence is below threshold
                       self.logger.debug(f"Confidence {confidence:.2f} for signal {signal} meets threshold.")
                  except Exception as e:
                       self.logger.error(f"Error during confidence filter check: {e}. Blocking entry.", exc_info=True)
                       return 0 # Block entry on filter error


        # --- EMA Trend Filter ---
        if self.trend_filter_enabled:
            try:
                # Use the EMA column name determined during initialization
                if self.ema_filter_col_name not in latest_features.columns:
                     self.logger.warning(f"Trend filter enabled, but EMA column '{self.ema_filter_col_name}' missing. Blocking trade.")
                     return 0

                # Get latest close price and EMA value (from the single row DataFrame)
                latest_ema = current_bar.get(self.ema_filter_col_name)

                if pd.isna(current_close) or pd.isna(latest_ema):
                     self.logger.warning("Trend filter cannot be applied due to NaN close or EMA value. Blocking trade.")
                     return 0

                # Determine trend state: 1 if close > EMA, -1 if close < EMA
                # Add a small tolerance (FLOAT_EPSILON) for floating point comparisons
                if current_close > latest_ema + FLOAT_EPSILON:
                    filter_state_int = 1
                elif current_close < latest_ema - FLOAT_EPSILON:
                    filter_state_int = -1
                else: # close is approximately equal to EMA
                    filter_state_int = 0 # Neutral trend state

                if (signal == 1 and filter_state_int != 1) or \
                   (signal == -1 and filter_state_int != -1):
                    self.logger.info(f"Trend filter blocked signal {signal} (Close={current_close:.4f}, EMA={latest_ema:.4f} -> Trend state: {filter_state_int}).")
                    return 0
                else:
                    self.logger.debug(f"Trend filter passed signal {signal} (Close={current_close:.4f}, EMA={latest_ema:.4f} -> Trend state: {filter_state_int}).")

            except Exception as e:
                 self.logger.error(f"Error applying trend filter: {e}. Blocking trade.", exc_info=True)
                 return 0

        # --- Add other entry filters here ---
        # Example: Volatility filter (e.g., block if ATR is too low/high)

        return signal # Signal passes all filters


    # MODIFIED: Removed _apply_signal_filters, as its logic is now part of _apply_entry_filters
    # and signal-based exits are handled directly in _execute_trade_logic based on raw signal.


    async def _execute_trade_logic(self, signal: int, latest_features: pd.DataFrame, latest_probabilities: Optional[pd.DataFrame]):
        """Handles trade execution based on signal, current position state, and filters."""
        self.logger.debug(f"Executing trade logic for signal: {signal}")

        await self._refresh_position_and_orders()

        # Check for and cancel orphaned orders
        if self.current_position is None and self.open_orders:
            self.logger.warning("Detected open orders but no open position. Cancelling orphaned orders.")
            await self._send_notification("WARNING: Detected orphaned orders. Cancelling...", level="warning")
            await self._cancel_open_orders()
            await self._refresh_position_and_orders()

        position_direction = self.current_position.get('direction') if self.current_position else None
        position_quantity = self.current_position.get('quantity') if self.current_position else 0.0

        # --- Scenario 1: No Position ---
        if position_direction is None:
            # Apply entry filters here, *before* attempting to open a position
            # Pass the latest features and probabilities to the filters
            filtered_signal = self._apply_entry_filters(signal, latest_features, latest_probabilities) # MODIFIED: Apply entry filters

            if filtered_signal == 1:
                # Add check for allowed long trades
                if self.allow_long_trades:
                    self.logger.info("Signal: LONG, Filters Passed. No position open. Attempting to open LONG.")
                    await self._open_position(side='buy')
                else:
                    self.logger.debug("Signal: LONG, Filters Passed. Long trades are DISABLED. Skipping entry.")
            elif filtered_signal == -1:
                # Add check for allowed short trades
                if self.allow_short_trades:
                    self.logger.info("Signal: SHORT, Filters Passed. No position open. Attempting to open SHORT.")
                    await self._open_position(side='sell')
                else:
                    self.logger.debug("Signal: SHORT, Filters Passed. Short trades are DISABLED. Skipping entry.")
            else: # filtered_signal is 0 (either raw signal was 0 or filters blocked it)
                self.logger.debug(f"Signal: {signal}, Filters Blocked. No position open. No action.")


        # --- Scenario 2: Long Position Open ---
        elif position_direction == 'long':
            if signal == 1:
                self.logger.debug("Signal: LONG. Already LONG. Holding position.")
            elif signal == -1:
                # Add check for allowed short trades (for reversal)
                if self.allow_short_trades:
                    self.logger.info("Signal: SHORT. Currently LONG. Attempting to REVERSE (close LONG, open SHORT).")
                    await self._close_position(side='sell', quantity=position_quantity, reason='signal_reverse')
                    await asyncio.sleep(1.0) # Small delay to allow close order to process
                    # After closing, we are flat. The next cycle's _execute_trade_logic will process
                    # the new signal (-1) and potentially open the short position, applying filters.
                    # No need to call _open_position directly here.
                else:
                    self.logger.debug("Signal: SHORT. Currently LONG. Short trades are DISABLED, cannot reverse. Holding LONG.")
            # Modified: Only close on neutral signal if exit_on_neutral_signal is True
            elif signal == 0:
                if self.exit_on_neutral_signal:
                    self.logger.info("Signal: NEUTRAL. Currently LONG. Exit on neutral is ENABLED. Attempting to CLOSE LONG.")
                    await self._close_position(side='sell', quantity=position_quantity, reason='signal_exit_neutral') # Use specific reason
                else:
                    self.logger.debug("Signal: NEUTRAL. Currently LONG. Exit on neutral is DISABLED. Holding position.")


        # --- Scenario 3: Short Position Open ---
        elif position_direction == 'short':
            if signal == 1:
                # Add check for allowed long trades (for reversal)
                if self.allow_long_trades:
                    self.logger.info("Signal: LONG. Currently SHORT. Attempting to REVERSE (close SHORT, open LONG).")
                    await self._close_position(side='buy', quantity=position_quantity, reason='signal_reverse')
                    await asyncio.sleep(1.0) # Small delay to allow close order to process
                     # After closing, we are flat. The next cycle's _execute_trade_logic will process
                    # the new signal (1) and potentially open the long position, applying filters.
                    # No need to call _open_position directly here.
                else:
                    self.logger.debug("Signal: LONG. Currently SHORT. Long trades are DISABLED, cannot reverse. Holding SHORT.")
            elif signal == -1:
                self.logger.debug("Signal: SHORT. Already SHORT. Holding position.")
            # Modified: Only close on neutral signal if exit_on_neutral_signal is True
            elif signal == 0:
                 if self.exit_on_neutral_signal:
                    self.logger.info("Signal: NEUTRAL. Currently SHORT. Exit on neutral is ENABLED. Attempting to CLOSE SHORT.")
                    await self._close_position(side='buy', quantity=position_quantity, reason='signal_exit_neutral') # Use specific reason
                 else:
                    self.logger.debug("Signal: NEUTRAL. Currently SHORT. Exit on neutral is DISABLED. Holding position.")


        # --- Check Max Holding Period ---
        # This check happens regardless of signal, but only if a position is open
        if self.current_position: # Check if a position is open
            # The max holding period is now dynamic based on volatility regime
            await self._check_max_holding_period(latest_features) # MODIFIED: Pass latest features for regime check


    def _apply_entry_filters(self, signal: int, latest_features: pd.DataFrame, latest_probabilities: Optional[pd.DataFrame]) -> int:
        """
        Applies configured filters (e.g., EMA trend, confidence threshold, volatility regime) to an entry signal.
        This is called when the bot is FLAT and receives a non-neutral signal.
        Filters are based on data from the latest bar.
        """
        # Signal is guaranteed to be non-zero here (1 or -1)
        if signal == 0: return 0 # Should not be called with signal 0, but defensive
        if latest_features.empty:
             self.logger.warning("Input data for entry filters is empty.")
             return 0

        current_bar = latest_features.iloc[-1] # Get the single row for the latest bar
        current_timestamp = latest_features.index[-1]
        current_close = current_bar['close']

        # --- Volatility Regime Filter (NEW) ---
        if self.volatility_regime_filter_enabled:
             try:
                  # Get the volatility regime for the current bar
                  # Use .get() with default pd.NA for robustness if column is missing
                  current_regime = current_bar.get(self.volatility_regime_col_name, pd.NA)

                  # Check if the regime value is valid (0, 1, or 2) and not NaN
                  if pd.isna(current_regime) or current_regime not in [0, 1, 2]:
                       self.logger.warning(f"Candle {current_timestamp}: Volatility regime value ({current_regime}) is invalid or NaN. Volatility regime filter blocks entry.")
                       return 0 # Block entry if regime is invalid

                  # Check if trading is allowed in this regime
                  # Use .get() with default False for robustness if regime key is missing (shouldn't happen with validation)
                  if not self.allow_trading_in_volatility_regime.get(current_regime, False):
                       self.logger.debug(f"Candle {current_timestamp}: Trading is not allowed in volatility regime {current_regime}. Volatility regime filter blocks signal {signal}.")
                       return 0 # Block entry if trading is not allowed in this regime
                  self.logger.debug(f"Candle {current_timestamp}: Trading is allowed in volatility regime {current_regime}. Passed volatility regime filter.")

             except KeyError:
                  # This should ideally be caught during feature engineering, but defensive here
                  self.logger.error(f"Volatility regime filter enabled, but column '{self.volatility_regime_col_name}' not found in latest features. Disabling filter and blocking trade.")
                  self.volatility_regime_filter_enabled = False # Disable filter permanently if column missing
                  return 0 # Block trade
             except Exception as e:
                  self.logger.error(f"Error during volatility regime filter check at candle {current_timestamp}: {e}. Blocking entry.", exc_info=True)
                  return 0 # Block entry on filter error


        # --- Confidence Threshold Filter ---
        if self.confidence_filter_enabled:
             # Determine the correct threshold based on the signal direction
             confidence_threshold = self.confidence_threshold_long if signal == 1 else self.confidence_threshold_short

             # If the threshold is 0 or less, the filter is effectively disabled for this direction
             if confidence_threshold > FLOAT_EPSILON:
                  if latest_probabilities is None or latest_probabilities.empty:
                       self.logger.warning("Confidence filter enabled but probabilities are missing or empty. Blocking entry.")
                       return 0 # Block entry if probabilities are missing

                  try:
                       # Get the probability for the predicted signal from the latest_probabilities DataFrame
                       # Use .get() with default 0.0 for robustness if a column is missing (shouldn't happen if ModelTrainer.predict_proba is correct)
                       confidence = latest_probabilities.iloc[-1].get(signal, 0.0)
                       if pd.isna(confidence):
                            self.logger.debug(f"Confidence score is NaN for signal {signal}. Confidence filter blocks entry.")
                            return 0 # Block entry if confidence is NaN
                       if confidence < confidence_threshold:
                            self.logger.debug(f"Confidence {confidence:.2f} for signal {signal} is below threshold {confidence_threshold:.2f}. Confidence filter blocks entry.")
                            return 0 # Block entry if confidence is below threshold
                       self.logger.debug(f"Confidence {confidence:.2f} for signal {signal} meets threshold.")
                  except Exception as e:
                       self.logger.error(f"Error during confidence filter check: {e}. Blocking entry.", exc_info=True)
                       return 0 # Block entry on filter error


        # --- EMA Trend Filter ---
        if self.trend_filter_enabled:
            try:
                # Use the EMA column name determined during initialization
                if self.ema_filter_col_name not in latest_features.columns:
                     self.logger.warning(f"Trend filter enabled, but EMA column '{self.ema_filter_col_name}' missing. Blocking trade.")
                     return 0

                # Get latest close price and EMA value (from the single row DataFrame)
                latest_ema = current_bar.get(self.ema_filter_col_name)

                if pd.isna(current_close) or pd.isna(latest_ema):
                     self.logger.warning("Trend filter cannot be applied due to NaN close or EMA value. Blocking trade.")
                     return 0

                # Determine trend state: 1 if close > EMA, -1 if close < EMA
                # Add a small tolerance (FLOAT_EPSILON) for floating point comparisons
                if current_close > latest_ema + FLOAT_EPSILON:
                    filter_state_int = 1
                elif current_close < latest_ema - FLOAT_EPSILON:
                    filter_state_int = -1
                else: # close is approximately equal to EMA
                    filter_state_int = 0 # Neutral trend state

                if (signal == 1 and filter_state_int != 1) or \
                   (signal == -1 and filter_state_int != -1):
                    self.logger.info(f"Trend filter blocked signal {signal} (Close={current_close:.4f}, EMA={latest_ema:.4f} -> Trend state: {filter_state_int}).")
                    return 0
                else:
                    self.logger.debug(f"Trend filter passed signal {signal} (Close={current_close:.4f}, EMA={latest_ema:.4f} -> Trend state: {filter_state_int}).")

            except Exception as e:
                 self.logger.error(f"Error applying trend filter: {e}. Blocking trade.", exc_info=True)
                 return 0

        # --- Add other entry filters here ---
        # Example: Volatility filter (e.g., block if ATR is too low/high)

        return signal # Signal passes all filters


    # MODIFIED: Removed _apply_signal_filters, as its logic is now part of _apply_entry_filters
    # and signal-based exits are handled directly in _execute_trade_logic based on raw signal.


    async def _execute_trade_logic(self, signal: int, latest_features: pd.DataFrame, latest_probabilities: Optional[pd.DataFrame]):
        """Handles trade execution based on signal, current position state, and filters."""
        self.logger.debug(f"Executing trade logic for signal: {signal}")

        await self._refresh_position_and_orders()

        # Check for and cancel orphaned orders
        if self.current_position is None and self.open_orders:
            self.logger.warning("Detected open orders but no open position. Cancelling orphaned orders.")
            await self._send_notification("WARNING: Detected orphaned orders. Cancelling...", level="warning")
            await self._cancel_open_orders()
            await self._refresh_position_and_orders()

        position_direction = self.current_position.get('direction') if self.current_position else None
        position_quantity = self.current_position.get('quantity') if self.current_position else 0.0

        # --- Scenario 1: No Position ---
        if position_direction is None:
            # Apply entry filters here, *before* attempting to open a position
            # Pass the latest features and probabilities to the filters
            filtered_signal = self._apply_entry_filters(signal, latest_features, latest_probabilities) # MODIFIED: Apply entry filters

            if filtered_signal == 1:
                # Add check for allowed long trades
                if self.allow_long_trades:
                    self.logger.info("Signal: LONG, Filters Passed. No position open. Attempting to open LONG.")
                    await self._open_position(side='buy')
                else:
                    self.logger.debug("Signal: LONG, Filters Passed. Long trades are DISABLED. Skipping entry.")
            elif filtered_signal == -1:
                # Add check for allowed short trades
                if self.allow_short_trades:
                    self.logger.info("Signal: SHORT, Filters Passed. No position open. Attempting to open SHORT.")
                    await self._open_position(side='sell')
                else:
                    self.logger.debug("Signal: SHORT, Filters Passed. Short trades are DISABLED. Skipping entry.")
            else: # filtered_signal is 0 (either raw signal was 0 or filters blocked it)
                self.logger.debug(f"Signal: {signal}, Filters Blocked. No position open. No action.")


        # --- Scenario 2: Long Position Open ---
        elif position_direction == 'long':
            if signal == 1:
                self.logger.debug("Signal: LONG. Already LONG. Holding position.")
            elif signal == -1:
                # Add check for allowed short trades (for reversal)
                if self.allow_short_trades:
                    self.logger.info("Signal: SHORT. Currently LONG. Attempting to REVERSE (close LONG, open SHORT).")
                    await self._close_position(side='sell', quantity=position_quantity, reason='signal_reverse')
                    await asyncio.sleep(1.0) # Small delay to allow close order to process
                    # After closing, we are flat. The next cycle's _execute_trade_logic will process
                    # the new signal (-1) and potentially open the short position, applying filters.
                    # No need to call _open_position directly here.
                else:
                    self.logger.debug("Signal: SHORT. Currently LONG. Short trades are DISABLED, cannot reverse. Holding LONG.")
            # Modified: Only close on neutral signal if exit_on_neutral_signal is True
            elif signal == 0:
                if self.exit_on_neutral_signal:
                    self.logger.info("Signal: NEUTRAL. Currently LONG. Exit on neutral is ENABLED. Attempting to CLOSE LONG.")
                    await self._close_position(side='sell', quantity=position_quantity, reason='signal_exit_neutral') # Use specific reason
                else:
                    self.logger.debug("Signal: NEUTRAL. Currently LONG. Exit on neutral is DISABLED. Holding position.")


        # --- Scenario 3: Short Position Open ---
        elif position_direction == 'short':
            if signal == 1:
                # Add check for allowed long trades (for reversal)
                if self.allow_long_trades:
                    self.logger.info("Signal: LONG. Currently SHORT. Attempting to REVERSE (close SHORT, open LONG).")
                    await self._close_position(side='buy', quantity=position_quantity, reason='signal_reverse')
                    await asyncio.sleep(1.0) # Small delay to allow close order to process
                     # After closing, we are flat. The next cycle's _execute_trade_logic will process
                    # the new signal (1) and potentially open the long position, applying filters.
                    # No need to call _open_position directly here.
                else:
                    self.logger.debug("Signal: LONG. Currently SHORT. Long trades are DISABLED, cannot reverse. Holding SHORT.")
            elif signal == -1:
                self.logger.debug("Signal: SHORT. Already SHORT. Holding position.")
            # Modified: Only close on neutral signal if exit_on_neutral_signal is True
            elif signal == 0:
                 if self.exit_on_neutral_signal:
                    self.logger.info("Signal: NEUTRAL. Currently SHORT. Exit on neutral is ENABLED. Attempting to CLOSE SHORT.")
                    await self._close_position(side='buy', quantity=position_quantity, reason='signal_exit_neutral') # Use specific reason
                 else:
                    self.logger.debug("Signal: NEUTRAL. Currently SHORT. Exit on neutral is DISABLED. Holding position.")


        # --- Check Max Holding Period ---
        # This check happens regardless of signal, but only if a position is open
        if self.current_position: # Check if a position is open
            # The max holding period is now dynamic based on volatility regime
            await self._check_max_holding_period(latest_features) # MODIFIED: Pass latest features for regime check


    async def _refresh_position_and_orders(self):
        """
        Fetches and updates internal state for open positions and orders.
        Detects if a position was closed externally (e.g., SL/TP hit) and updates capital.
        """
        self.logger.debug("Refreshing position and order state from exchange...")
        try:
            position_before_refresh = copy.deepcopy(self.current_position)
            local_entry_time_dt = None
            if self.current_position:
                entry_time_val = self.current_position.get('entryTime')
                if isinstance(entry_time_val, datetime):
                    local_entry_time_dt = entry_time_val
                elif isinstance(entry_time_val, str):
                    try:
                        local_entry_time_dt = pd.Timestamp(entry_time_val).to_pydatetime()
                        if local_entry_time_dt.tzinfo is None or local_entry_time_dt.tzinfo.utcoffset(local_entry_time_dt) is None:
                           local_entry_time_dt = local_entry_time_dt.replace(tzinfo=timezone.utc)
                        else:
                           local_entry_time_dt = local_entry_time_dt.astimezone(timezone.utc)
                    except Exception as e_parse_local:
                        self.logger.warning(f"Could not parse local_entry_time string '{entry_time_val}': {e_parse_local}")
                elif entry_time_val is not None: # pd.Timestamp can handle some numeric types too
                    try:
                        local_entry_time_dt = pd.Timestamp(entry_time_val).to_pydatetime()
                        if local_entry_time_dt.tzinfo is None or local_entry_time_dt.tzinfo.utcoffset(local_entry_time_dt) is None:
                           local_entry_time_dt = local_entry_time_dt.replace(tzinfo=timezone.utc)
                        else:
                           local_entry_time_dt = local_entry_time_dt.astimezone(timezone.utc)
                    except Exception as e_parse_local_num:
                         self.logger.warning(f"Could not parse local_entry_time value '{entry_time_val}' of type {type(entry_time_val)}: {e_parse_local_num}")


            open_positions = await self.exchange_adapter.get_open_positions(symbol=self.symbol)
            fetched_position_data = open_positions[0] if open_positions else None

            if fetched_position_data:
                 for key in ['direction', 'quantity', 'entryPrice', 'entryTime', 'unrealizedPnl', 'leverage', 'entryMargin', 'liquidationPrice']:
                      if key not in fetched_position_data: fetched_position_data[key] = None

                 exchange_entry_time_raw = fetched_position_data.get('entryTime')
                 exchange_entry_time_dt = None
                 if exchange_entry_time_raw is not None:
                      try:
                           if isinstance(exchange_entry_time_raw, (int, float)): # Typically ms timestamp
                                exchange_entry_time_dt = pd.Timestamp(exchange_entry_time_raw, unit='ms', tz='UTC').to_pydatetime()
                           elif isinstance(exchange_entry_time_raw, str):
                                exchange_entry_time_dt = pd.Timestamp(exchange_entry_time_raw).to_pydatetime()
                           elif isinstance(exchange_entry_time_raw, datetime): # Already datetime
                                exchange_entry_time_dt = exchange_entry_time_raw
                           else: # pd.Timestamp might handle other types
                                exchange_entry_time_dt = pd.Timestamp(exchange_entry_time_raw).to_pydatetime()

                           # Ensure timezone is UTC
                           if exchange_entry_time_dt.tzinfo is None or exchange_entry_time_dt.tzinfo.utcoffset(exchange_entry_time_dt) is None:
                               exchange_entry_time_dt = exchange_entry_time_dt.replace(tzinfo=timezone.utc)
                           else:
                               exchange_entry_time_dt = exchange_entry_time_dt.astimezone(timezone.utc)
                           fetched_position_data['entryTime'] = exchange_entry_time_dt
                           self.logger.debug(f"Exchange 'entryTime' parsed to: {exchange_entry_time_dt}")
                      except Exception as e_ts_exchange:
                           self.logger.warning(f"Fetched 'entryTime' '{exchange_entry_time_raw}' from exchange is not parsable: {e_ts_exchange}. Using local if available.")
                           fetched_position_data['entryTime'] = local_entry_time_dt
                 elif local_entry_time_dt:
                      fetched_position_data['entryTime'] = local_entry_time_dt
                      self.logger.debug(f"Used preserved local 'entryTime': {local_entry_time_dt}")
                 else:
                      self.logger.warning(f"Position data for {self.symbol} missing 'entryTime' from exchange and no valid local entry time. Max holding check may fail.")
                      fetched_position_data['entryTime'] = None

                 if position_before_refresh: # Use 'position_before_refresh' for bot-tracked IDs
                     fetched_position_data['entryOrderId'] = position_before_refresh.get('entryOrderId')
                     fetched_position_data['slOrderId'] = position_before_refresh.get('slOrderId')
                     fetched_position_data['tpOrderId'] = position_before_refresh.get('tpOrderId')
                 else: # No prior position, ensure these keys exist if fetched_position_data is new
                     fetched_position_data.setdefault('entryOrderId', None)
                     fetched_position_data.setdefault('slOrderId', None)
                     fetched_position_data.setdefault('tpOrderId', None)


                 self.current_position = fetched_position_data
            else:
                 self.current_position = None

            self.open_orders = await self.exchange_adapter.get_open_orders(symbol=self.symbol)

            if position_before_refresh and not self.current_position:
                self.logger.info(f"Position {position_before_refresh.get('direction')} {position_before_refresh.get('quantity', 0.0):.8f} "
                                 f"for {self.symbol} (Entry Order ID: {position_before_refresh.get('entryOrderId')}) "
                                 f"appears to have been closed externally (e.g., SL/TP hit or liquidation).")

                sl_order_id = position_before_refresh.get('slOrderId')
                tp_order_id = position_before_refresh.get('tpOrderId')
                exit_reason = "unknown_external_close"
                filled_order_details = None
                final_exit_order_id = None

                if sl_order_id:
                    try:
                        self.logger.debug(f"Checking status of potential SL order: {sl_order_id}")
                        order_info = await self.exchange_adapter.get_order_info(self.symbol, str(sl_order_id))
                        if order_info and order_info.get('status') == 'FILLED':
                            self.logger.info(f"Stop Loss order {sl_order_id} confirmed FILLED.")
                            filled_order_details = order_info
                            exit_reason = "stop_loss_hit"
                            final_exit_order_id = str(sl_order_id)
                    except Exception as e_sl:
                        self.logger.warning(f"Could not get SL order {sl_order_id} info during external close check: {e_sl}")

                if not filled_order_details and tp_order_id:
                    try:
                        self.logger.debug(f"Checking status of potential TP order: {tp_order_id}")
                        order_info = await self.exchange_adapter.get_order_info(self.symbol, str(tp_order_id))
                        if order_info and order_info.get('status') == 'FILLED':
                            self.logger.info(f"Take Profit order {tp_order_id} confirmed FILLED.")
                            filled_order_details = order_info
                            exit_reason = "take_profit_hit"
                            final_exit_order_id = str(tp_order_id)
                    except Exception as e_tp:
                        self.logger.warning(f"Could not get TP order {tp_order_id} info during external close check: {e_tp}")

                if filled_order_details:
                    executed_qty_close = float(filled_order_details.get('executedQty', 0.0))
                    avg_exit_price_str = filled_order_details.get('avgPrice', '0.0')
                    avg_exit_price = 0.0
                    try:
                        avg_exit_price = float(avg_exit_price_str)
                    except ValueError:
                        self.logger.warning(f"Could not parse avgPrice '{avg_exit_price_str}' to float for external close.")

                    if executed_qty_close > FLOAT_EPSILON and (avg_exit_price <= FLOAT_EPSILON or pd.isna(avg_exit_price)):
                        cum_quote_str = filled_order_details.get('cumQuote')
                        if cum_quote_str:
                            try:
                                cum_quote_val = float(cum_quote_str)
                                if cum_quote_val > FLOAT_EPSILON:
                                    avg_exit_price = cum_quote_val / executed_qty_close
                                    self.logger.info(f"Calculated avg_exit_price ({avg_exit_price:.8f}) from cumQuote ({cum_quote_val}) and executedQty ({executed_qty_close:.8f}) for external close.")
                            except ValueError:
                                self.logger.warning(f"Could not parse cumQuote '{cum_quote_str}' to float for external close avg_exit_price calculation.")
                        else:
                            self.logger.warning("avgPrice is invalid and cumQuote is missing for PnL calculation on external close.")

                    original_pos_qty = float(position_before_refresh.get('quantity', 0.0))
                    if abs(executed_qty_close - original_pos_qty) > (FLOAT_EPSILON + original_pos_qty * 0.01) :
                        self.logger.warning(f"Externally closed quantity ({executed_qty_close:.8f}) "
                                            f"differs from original position quantity ({original_pos_qty:.8f}). "
                                            f"Using exchange-reported executed quantity for PnL.")

                    if executed_qty_close > FLOAT_EPSILON and avg_exit_price > FLOAT_EPSILON:
                        self.logger.info(f"Processing external close: Qty={executed_qty_close:.8f}, Price={avg_exit_price:.8f}, Reason={exit_reason}, OrderID={final_exit_order_id}")
                        self._update_state_after_close(
                            closed_position_details=position_before_refresh,
                            executed_qty_close=executed_qty_close,
                            avg_exit_price=avg_exit_price,
                            reason=exit_reason,
                            exit_order_id=final_exit_order_id
                        )
                        await self._send_notification(
                            f"{exit_reason.replace('_', ' ').upper()} for {self.symbol}. "
                            f"Closed {executed_qty_close:.8f} @ {avg_exit_price:.8f}. Capital updated.",
                            level="info"
                        )
                        asyncio.create_task(self._async_save_current_state())
                    else:
                        self.logger.error(f"External close for position (Entry Order ID: {position_before_refresh.get('entryOrderId')}) "
                                          f"detected (Reason: {exit_reason}, Trigger Order: {final_exit_order_id}), "
                                          f"but could not retrieve/calculate valid fill details (Qty: {executed_qty_close}, Price: {avg_exit_price}). "
                                          "Capital NOT updated. Manual check required.")
                        await self._send_notification(
                            f"CRITICAL: {self.symbol} position (Entry: {position_before_refresh.get('entryOrderId')}) "
                            f"closed externally (Reason: {exit_reason}, Trigger: {final_exit_order_id}), "
                            "but FAILED to get valid fill details. Capital NOT updated. MANUAL CHECK!",
                            level="critical"
                        )
                else:
                    self.logger.warning(f"Position (Entry Order ID: {position_before_refresh.get('entryOrderId')}) "
                                        f"closed externally for {self.symbol}, but could not confirm SL or TP fill. "
                                        "This might be due to liquidation or manual intervention. Capital update SKIPPED for this event to avoid inaccuracies.")
                    await self._send_notification(
                        f"WARNING: {self.symbol} position (Entry: {position_before_refresh.get('entryOrderId')}) "
                        f"closed externally, but SL/TP fill NOT confirmed. Capital NOT updated. Check for liquidation/manual action.",
                        level="warning"
                    )

            if self.current_position:
                 self.logger.debug(f"Refreshed Position: {self.current_position.get('direction')} {self.current_position.get('quantity'):.8f} @ Entry {self.current_position.get('entryPrice', 0.0):.8f}")
                 self.logger.debug(f"Current Position EntryTime: {self.current_position.get('entryTime')}")
            else:
                 self.logger.debug("Refreshed Position: None")
            self.logger.debug(f"Refreshed Open Orders: {len(self.open_orders)} found.")

        except Exception as e:
            self.logger.error(f"Error refreshing position/order state: {e}", exc_info=True)
            await self._send_notification(f"ERROR: Failed to refresh position/order state: {e}", level="error")

    async def _check_max_holding_period(self, latest_features: pd.DataFrame):
        """
        Checks if the current position has exceeded the max holding period,
        using the volatility regime-specific setting if enabled.
        """
        if not self.current_position: return # No position to check

        # --- Determine Max Holding Period for the CURRENT Regime (NEW) ---
        # Get the volatility regime for the latest bar
        current_regime = latest_features.iloc[-1].get(self.volatility_regime_col_name, pd.NA)

        max_holding_for_this_trade = self.max_holding_period_bars # Default to global config

        if self.volatility_regime_filter_enabled and pd.notna(current_regime) and current_regime in [0, 1, 2]:
             regime_holding = self.volatility_regime_max_holding_bars.get(current_regime)
             if regime_holding is not None and (isinstance(regime_holding, int) and regime_holding > 0):
                  max_holding_for_this_trade = regime_holding
                  self.logger.debug(f"Using regime-specific max holding: {max_holding_for_this_trade} bars (Regime: {current_regime})")
             elif regime_holding is None:
                  max_holding_for_this_trade = None # No time limit for this regime
                  self.logger.debug(f"No time limit for current regime {current_regime}.")
             else:
                  self.logger.warning(f"Invalid max holding period ({regime_holding}) configured for current regime {current_regime}. Using default ({self.max_holding_period_bars}).")
        elif self.volatility_regime_filter_enabled:
             self.logger.warning(f"Volatility regime ({current_regime}) invalid for determining max holding. Using default ({self.max_holding_period_bars}).")

        # If max_holding_for_this_trade is None, there's no time limit for this trade
        if max_holding_for_this_trade is None:
             self.logger.debug("Max holding period check skipped: No time limit configured for current regime or filter disabled/invalid.")
             return # Exit if no time limit is set for the current regime or globally

        # --- Proceed with time limit check using the determined max holding ---
        entry_time_value = self.current_position.get('entryTime')
        entry_time_dt: Optional[datetime] = None

        if isinstance(entry_time_value, datetime):
            entry_time_dt = entry_time_value
        elif isinstance(entry_time_value, str):
            try:
                entry_time_dt = pd.Timestamp(entry_time_value).to_pydatetime()
            except Exception as e_parse:
                self.logger.warning(f"Cannot check max holding period: Could not parse 'entryTime' string '{entry_time_value}': {e_parse}.")
                return
        elif entry_time_value is not None: # Try to parse if it's some other type (e.g. pd.Timestamp from loaded state)
             try:
                  entry_time_dt = pd.Timestamp(entry_time_value).to_pydatetime()
             except Exception as e_parse_other:
                  self.logger.warning(f"Cannot check max holding period: Could not parse 'entryTime' value '{entry_time_value}' of type {type(entry_time_value)}: {e_parse_other}.")
                  return
        else: # entry_time_value is None
            self.logger.warning("Cannot check max holding period: 'entryTime' is missing.")
            return

        # Ensure entry_time_dt is timezone-aware (UTC)
        if entry_time_dt.tzinfo is None or entry_time_dt.tzinfo.utcoffset(entry_time_dt) is None:
            entry_time_dt = entry_time_dt.replace(tzinfo=timezone.utc)
        else:
            entry_time_dt = entry_time_dt.astimezone(timezone.utc)

        try:
            current_time_utc = datetime.now(timezone.utc)
            interval_timedelta = self._parse_interval_to_timedelta(self.interval) # Assuming this method exists and works
            if not interval_timedelta or interval_timedelta.total_seconds() <= 0:
                 self.logger.warning("Cannot check max holding period: Invalid interval for timedelta.")
                 return

            holding_duration = current_time_utc - entry_time_dt
            # Calculate holding bars based on the time difference and interval duration
            holding_bars = math.floor(holding_duration.total_seconds() / interval_timedelta.total_seconds())

            # Check if holding bars meets or exceeds the determined max holding period
            if holding_bars >= max_holding_for_this_trade:
                 self.logger.info(f"Max holding period ({max_holding_for_this_trade} bars) reached ({holding_bars} held). Closing.")
                 await self._send_notification("INFO: Max holding period reached. Closing position.", level="info")
                 close_side = 'sell' if self.current_position.get('direction') == 'long' else 'buy'
                 await self._close_position(side=close_side, quantity=self.current_position.get('quantity'), reason='max_holding')
        except Exception as e:
             self.logger.error(f"Error checking max holding period: {e}", exc_info=True)


    async def _open_position(self, side: str):
        """Calculates size, places market entry order, and corresponding SL/TP orders."""
        if side not in ['buy', 'sell']:
            self.logger.error(f"Invalid side '{side}' for opening position.")
            return
        # Direction check is now done in _execute_trade_logic before calling _open_position

        if self.current_position:
            self.logger.warning(f"Attempted to open {side} position, but already in a {self.current_position.get('direction')} position. Skipping.")
            return

        self.logger.info(f"--- Attempting to Open {side.upper()} Position ---")
        order_placed = False
        sl_tp_placed = False
        entry_details = {} # Initialize to ensure it's always defined

        try:
            current_price = await self.exchange_adapter.get_latest_price(self.symbol)
            if not current_price or current_price <= 0: raise ValueError("Could not get valid current price.")

            stop_loss_price, take_profit_price = await self._calculate_sl_tp_prices(side, current_price)
            if not stop_loss_price: raise ValueError("Failed to calculate valid Stop Loss price.")

            position_quantity = await self._calculate_position_size(side, current_price, stop_loss_price)
            if not position_quantity or position_quantity <= 0: raise ValueError(f"Calculated position size ({position_quantity}) invalid.")

            estimated_liq_price = await self._estimate_liquidation_price(side, current_price)
            if estimated_liq_price and not self._is_sl_safe_from_liquidation(side, stop_loss_price, estimated_liq_price):
                 self.logger.critical(f"CRITICAL RISK: SL ({stop_loss_price:.8f}) too close to Liq ({estimated_liq_price:.8f}). ABORTING TRADE.")
                 await self._send_notification(f"CRITICAL RISK: SL ({stop_loss_price:.8f}) too close to Liq ({estimated_liq_price:.8f}). Trade Aborted.", level="critical")
                 return
            self.logger.info(f"SL ({stop_loss_price:.8f}) safe relative to estimated Liq ({estimated_liq_price:.8f}). Proceeding.")

            entry_order_result = await self.exchange_adapter.place_market_order(
                symbol=self.symbol, side='BUY' if side == 'buy' else 'SELL', quantity=position_quantity, reduce_only=False
            )
            if not entry_order_result or entry_order_result.get('status') in ['REJECTED', 'EXPIRED', 'CANCELED']:
                raise OrderExecutionError(f"Market entry order failed or rejected. Result: {entry_order_result}")
            order_placed = True
            entry_order_id = entry_order_result.get('orderId')
            if entry_order_id is None:
                self.logger.critical(f"Market entry order placed but NO Order ID returned by exchange. Result: {entry_order_result}. Tracking will be impaired.")
            else:
                self.logger.info(f"Market entry order initiated. Order ID: {entry_order_id}")


            await asyncio.sleep(1.5) # Wait for order to likely fill
            # Try to get filled order details, but have fallbacks
            filled_order_details = None
            if entry_order_id: # Only try if we have an ID
                try:
                    filled_order_details = await self.exchange_adapter.get_order_info(self.symbol, str(entry_order_id)) # Ensure ID is string
                except Exception as e_get_order:
                    self.logger.warning(f"Could not fetch entry order {entry_order_id} details: {e_get_order}. Will use estimates.")


            if not filled_order_details or filled_order_details.get('status') != 'FILLED':
                 self.logger.warning(f"Entry order {entry_order_id or 'N/A'} not confirmed FILLED. Status: {filled_order_details.get('status') if filled_order_details else 'N/A'}. Using estimated price/qty.")
                 current_price_fallback = await self.exchange_adapter.get_latest_price(self.symbol)
                 avg_entry_price = current_price_fallback if current_price_fallback and current_price_fallback > 0 else current_price
                 executed_qty = position_quantity # Use the requested quantity as fallback
                 entry_fee = (executed_qty * avg_entry_price) * self.trading_fee_rate
            else:
                 executed_qty = float(filled_order_details.get('executedQty', 0.0))
                 avg_entry_price_str = filled_order_details.get('avgPrice', '0.0')
                 avg_entry_price = float(avg_entry_price_str) if avg_entry_price_str and float(avg_entry_price_str) > FLOAT_EPSILON else 0.0

                 if avg_entry_price <= 0 and executed_qty > 0: # If avgPrice is bad, try cumQuote
                     cum_quote_str = filled_order_details.get('cumQuote') # Should be cumQuote for entry, not exit
                     if cum_quote_str:
                         try:
                             cum_quote_val = float(cum_quote_str)
                             if cum_quote_val > FLOAT_EPSILON:
                                 avg_entry_price = cum_quote_val / executed_qty
                                 self.logger.info(f"Calculated avg_entry_price ({avg_entry_price:.8f}) from cumQuote and executedQty for entry.")
                         except ValueError:
                             self.logger.warning(f"Could not parse cumQuote '{cum_quote_str}' for entry avgPrice calculation.")

                 if executed_qty <= FLOAT_EPSILON or avg_entry_price <= FLOAT_EPSILON:
                     raise OrderExecutionError(f"Entry order {entry_order_id} FILLED but invalid qty/price (Qty: {executed_qty}, Price: {avg_entry_price}).")

                 entry_fee = (executed_qty * avg_entry_price) * self.trading_fee_rate
                 self.logger.info(f"Entry order FILLED. Qty: {executed_qty:.8f}, Price: {avg_entry_price:.8f}, Fee: {entry_fee:.4f}")

            # Ensure entryTime is a datetime object for Parquet compatibility
            current_utc_time = datetime.now(timezone.utc)

            entry_details = {
                 'symbol': self.symbol, 'direction': side, 'quantity': executed_qty,
                 'entryPrice': avg_entry_price,
                 'entryTime': current_utc_time, # MODIFIED: Store as datetime object
                 'entryOrderId': str(entry_order_id) if entry_order_id is not None else None,
                 'entryFee': entry_fee, 'status': 'open',
                 'slOrderId': None, # Initialize, will be set by _place_sl_tp_orders
                 'tpOrderId': None  # Initialize, will be set by _place_sl_tp_orders
            }
            self.current_position = entry_details # Set current_position so _place_sl_tp_orders can update it

            sl_tp_placed = await self._place_sl_tp_orders(side, executed_qty, stop_loss_price, take_profit_price)
            if not sl_tp_placed: # _place_sl_tp_orders returns True if SL is placed, TP is optional
                 self.logger.critical("Failed to place mandatory Stop Loss order after entry. Closing position.")
                 await self._send_notification("CRITICAL: Failed to place SL order. Closing position.", level="critical")
                 close_side_emergency = 'sell' if side == 'buy' else 'buy'
                 await self._close_position(close_side_emergency, executed_qty, 'sl_placement_failure', is_emergency=True)
                 return

            self.internal_capital -= entry_fee
            await self._refresh_position_and_orders()
            self.logger.info(f"--- Successfully Opened {side.upper()} Position ---")
            self.logger.info(f"Position: {self.current_position.get('quantity', 0.0):.8f} {self.symbol} @ {self.current_position.get('entryPrice', 0.0):.8f}")
            self.logger.info(f"SL Order ID: {self.current_position.get('slOrderId')}, TP Order ID: {self.current_position.get('tpOrderId')}")
            self.logger.info(f"Internal Capital after fee: {self.internal_capital:.2f}")
            await self._send_notification(f"OPENED {side.upper()} {executed_qty:.8f} @ {avg_entry_price:.8f}. Capital: {self.internal_capital:.2f}", level="info")

            asyncio.create_task(self._async_save_current_state())

        except (ValueError, OrderExecutionError) as e:
             self.logger.error(f"Failed to open {side.upper()} position: {e}", exc_info=True)
             await self._send_notification(f"ERROR: Failed to open {side.upper()} position: {e}", level="error")
             if order_placed and not sl_tp_placed:
                  self.logger.error("Entry order might be open without SL/TP. Manual check required! Attempting emergency close.")
                  await self._send_notification("CRITICAL: Entry order might be open without SL/TP. Attempting emergency close. Manual check required!", level="critical")
                  qty_to_close_emergency = entry_details.get('quantity', position_quantity if 'position_quantity' in locals() and position_quantity > 0 else 0)
                  if qty_to_close_emergency > 0:
                       close_side_emergency = 'sell' if side == 'buy' else 'buy'
                       await self._close_position(close_side_emergency, qty_to_close_emergency, 'entry_sl_tp_failure_cleanup', is_emergency=True)
                  else:
                       self.logger.error("Cannot perform emergency close: unknown quantity.")
             self.current_position = None
        except Exception as e:
             self.logger.critical(f"Unexpected critical error during open_position: {e}", exc_info=True)
             await self._send_notification(f"CRITICAL ERROR during open_position: {e}", level="critical")
             self.current_position = None

    async def _calculate_sl_tp_prices(self, side: str, current_price: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculates SL and TP prices based on volatility or fixed percentages."""
        self.logger.debug(f"Calculating SL/TP for {side.upper()} entry at {current_price:.8f}")
        sl_pct_fraction = self.fixed_stop_loss_fraction
        tp_pct_fraction = self.fixed_take_profit_fraction

        if self.volatility_adjustment_enabled:
            try:
                # Need enough data in the buffer to calculate ATR
                min_required_for_atr = self.volatility_window_bars + 1 # ATR needs window + 1 bars
                if len(self.data_buffer) >= min_required_for_atr:
                    # Process a slice to get latest ATR efficiently
                    # Use a slice that includes the ATR window plus a few extra bars
                    vol_data = self.data_buffer.tail(self.volatility_window_bars + 5).copy()
                    vol_features = self.feature_engineer.process(vol_data)
                    # Ensure the ATR column exists in the featured data
                    if self.atr_col_name_volatility in vol_features.columns:
                         latest_atr = vol_features[self.atr_col_name_volatility].iloc[-1]
                    else:
                         latest_atr = pd.NA
                         self.logger.warning(f"ATR column '{self.atr_col_name_volatility}' not found in featured data for volatility adjustment.")


                    if pd.notna(latest_atr) and latest_atr > 0:
                        dynamic_sl_pct = (self.alpha_stop_loss * latest_atr) / current_price
                        dynamic_tp_pct = (self.alpha_take_profit * latest_atr) / current_price
                        sl_pct_fraction = max(self.fixed_stop_loss_fraction, dynamic_sl_pct)
                        tp_pct_fraction = max(self.fixed_take_profit_fraction, dynamic_tp_pct)
                        self.logger.debug(f"Dynamic SL/TP based on ATR {latest_atr:.8f}: SL%={dynamic_sl_pct:.4f}, TP%={dynamic_tp_pct:.4f}")
                    else:
                        self.logger.warning(f"Latest ATR ({latest_atr}) invalid. Using fixed SL/TP percentages.")
                else:
                    self.logger.warning(f"Insufficient data ({len(self.data_buffer)}) for ATR calculation ({min_required_for_atr} needed). Using fixed SL/TP percentages.")
            except Exception as e:
                self.logger.error(f"Error calculating dynamic SL/TP: {e}. Using fixed percentages.", exc_info=True)

        self.logger.debug(f"Final SL/TP Percentages: SL={sl_pct_fraction:.4f}, TP={tp_pct_fraction:.4f}")

        if side == 'buy':
            stop_loss_price = current_price * (1 - sl_pct_fraction)
            take_profit_price = current_price * (1 + tp_pct_fraction)
        else:
            stop_loss_price = current_price * (1 + sl_pct_fraction)
            take_profit_price = current_price * (1 - tp_pct_fraction)

        adjusted_sl = await self._adjust_price_for_exchange(stop_loss_price)
        adjusted_tp = await self._adjust_price_for_exchange(take_profit_price)

        if not adjusted_sl or adjusted_sl <= 0:
             self.logger.error(f"Adjusted Stop Loss price ({adjusted_sl}) invalid.")
             adjusted_sl = None
        if not adjusted_tp or adjusted_tp <= 0:
             self.logger.warning(f"Adjusted Take Profit price ({adjusted_tp}) invalid. Proceeding without TP.")
             adjusted_tp = None

        sl_str = f"{adjusted_sl:.8f}" if adjusted_sl is not None else "N/A"
        tp_str = f"{adjusted_tp:.8f}" if adjusted_tp is not None else "N/A"
        self.logger.info(f"Calculated Prices: SL={sl_str}, TP={tp_str}")
        return adjusted_sl, adjusted_tp


    async def _calculate_position_size(self, side: str, current_price: float, stop_loss_price: float) -> Optional[float]:
        """Calculates position size based on risk, capital, leverage, and SL distance."""
        self.logger.debug(f"Calculating position size for {side.upper()} entry at {current_price:.8f}, SL at {stop_loss_price:.8f}")
        capital_to_risk = self.internal_capital * self.risk_per_trade_fraction
        stop_loss_distance = abs(current_price - stop_loss_price)
        if stop_loss_distance <= FLOAT_EPSILON:
            self.logger.error("Stop loss distance is zero or negative.")
            return None

        risk_based_quantity = capital_to_risk / stop_loss_distance
        initial_margin_rate = 1.0 / self.leverage
        effective_cost_rate = initial_margin_rate + self.trading_fee_rate
        if effective_cost_rate <= FLOAT_EPSILON:
             max_allowed_quantity = float('inf')
        else:
             max_position_value_usd = self.internal_capital / effective_cost_rate
             max_allowed_quantity = max_position_value_usd / current_price if current_price > 0 else 0

        final_raw_quantity = min(risk_based_quantity, max_allowed_quantity)
        adjusted_quantity = await self._adjust_quantity_for_exchange(final_raw_quantity)

        if not adjusted_quantity or adjusted_quantity <= FLOAT_EPSILON:
             self.logger.error(f"Final adjusted quantity ({adjusted_quantity}) invalid or zero.")
             return None

        notional_value = adjusted_quantity * current_price
        required_margin = notional_value / self.leverage
        estimated_fee = notional_value * self.trading_fee_rate
        required_capital = required_margin + estimated_fee

        if required_capital > self.internal_capital + FLOAT_EPSILON:
             self.logger.error(f"Insufficient internal capital ({self.internal_capital:.2f}) for size ({adjusted_quantity:.8f}). Required: {required_capital:.2f}.")
             await self._send_notification(f"ERROR: Insufficient capital ({self.internal_capital:.2f}) for trade. Required: {required_capital:.2f}", level="error")
             return None

        self.logger.info(f"Calculated Position Size: {adjusted_quantity:.8f} {self.symbol}")
        self.logger.debug(f"  Notional: {notional_value:.2f}, Margin: {required_margin:.2f}, Est. Fee: {estimated_fee:.4f}")
        return adjusted_quantity


    async def _estimate_liquidation_price(self, side: str, entry_price: float) -> Optional[float]:
        """Estimates liquidation price."""
        self.logger.debug(f"Estimating liquidation price for {side.upper()} entry at {entry_price:.8f}")
        try:
            # Use maintenance margin rate from config
            maint_margin_rate = self.config.get('maintenance_margin_rate', 0.004) # Default to 0.4% if not in config
            initial_margin_rate = 1.0 / self.leverage
            if initial_margin_rate <= maint_margin_rate:
                self.logger.warning("Initial margin rate <= maintenance margin rate. Liq likely immediate.")
                return entry_price * (1 - (1 if side == 'buy' else -1) * FLOAT_EPSILON)

            if side == 'buy':
                liq_price = entry_price * (1 - (initial_margin_rate - maint_margin_rate))
            else:
                liq_price = entry_price * (1 + (initial_margin_rate - maint_margin_rate))

            adjusted_liq_price = await self._adjust_price_for_exchange(liq_price)
            if adjusted_liq_price and adjusted_liq_price > 0:
                 self.logger.debug(f"Estimated Liquidation Price: {adjusted_liq_price:.8f}")
                 return adjusted_liq_price
            else:
                 self.logger.warning(f"Estimated liquidation price calculation invalid ({adjusted_liq_price}).")
                 return None
        except Exception as e:
            self.logger.error(f"Error estimating liquidation price: {e}", exc_info=True)
            return None


    def _is_sl_safe_from_liquidation(self, side: str, stop_loss_price: float, liquidation_price: float) -> bool:
        """Checks if the SL price is safely distanced from the liquidation price."""
        if not stop_loss_price or not liquidation_price:
             self.logger.warning("Cannot check SL safety: Invalid SL or Liquidation price.")
             return False
        safety_buffer = liquidation_price * self.min_liq_distance_fraction
        if side == 'buy':
            safe_sl_level = liquidation_price + safety_buffer
            is_safe = stop_loss_price > safe_sl_level - FLOAT_EPSILON # Use tolerance
            if not is_safe: self.logger.warning(f"Long SL check: SL ({stop_loss_price:.8f}) NOT above Liq ({liquidation_price:.8f}) + Buffer ({safety_buffer:.8f}) = {safe_sl_level:.8f}")
        else:
            safe_sl_level = liquidation_price - safety_buffer
            is_safe = stop_loss_price < safe_sl_level + FLOAT_EPSILON # Use tolerance
            if not is_safe: self.logger.warning(f"Short SL check: SL ({stop_loss_price:.8f}) NOT below Liq ({liquidation_price:.8f}) - Buffer ({safety_buffer:.8f}) = {safe_sl_level:.8f}")
        return is_safe


    async def _place_sl_tp_orders(self, side: str, quantity: float, stop_loss_price: float, take_profit_price: Optional[float]) -> bool:
        """Places STOP_MARKET (SL) and TAKE_PROFIT_MARKET (TP) orders."""
        self.logger.info(f"Placing SL/TP orders for {side.upper()} position, Qty: {quantity:.8f}")
        sl_tp_side = 'SELL' if side == 'buy' else 'BUY'
        sl_success = False
        tp_success = True # Assume TP success if not placing one

        # --- Place Stop Loss (Mandatory) ---
        try:
            self.logger.info(f"Placing STOP_MARKET (SL): Side={sl_tp_side}, Qty={quantity:.8f}, StopPrice={stop_loss_price:.8f}")
            sl_order_result = await self.exchange_adapter.place_stop_market_order(
                symbol=self.symbol, side=sl_tp_side, quantity=quantity, stop_price=stop_loss_price, reduce_only=True
            )
            if not sl_order_result or sl_order_result.get('status') in ['REJECTED', 'EXPIRED', 'CANCELED']:
                 raise OrderExecutionError(f"Stop Loss order placement failed or rejected. Result: {sl_order_result}")
            self.logger.info(f"Stop Loss order placed successfully. ID: {sl_order_result.get('orderId')}")
            if self.current_position: self.current_position['slOrderId'] = sl_order_result.get('orderId')
            sl_success = True
        except Exception as e:
            self.logger.error(f"Failed to place Stop Loss order: {e}", exc_info=True)
            sl_success = False

        # --- Place Take Profit (Optional) ---
        if take_profit_price and sl_success:
            try:
                self.logger.info(f"Placing TAKE_PROFIT_MARKET (TP): Side={sl_tp_side}, Qty={quantity:.8f}, StopPrice={take_profit_price:.8f}")
                tp_order_result = await self.exchange_adapter.place_take_profit_market_order(
                    symbol=self.symbol, side=sl_tp_side, quantity=quantity, stop_price=take_profit_price, reduce_only=True
                )
                if not tp_order_result or tp_order_result.get('status') in ['REJECTED', 'EXPIRED', 'CANCELED']:
                     self.logger.warning(f"Take Profit order placement failed or rejected. Result: {tp_order_result}")
                     tp_success = False
                else:
                     self.logger.info(f"Take Profit order placed successfully. ID: {tp_order_result.get('orderId')}")
                     if self.current_position: self.current_position['tpOrderId'] = tp_order_result.get('orderId')
                     tp_success = True
            except Exception as e:
                self.logger.warning(f"Failed to place Take Profit order: {e}", exc_info=True)
                tp_success = False
        elif not take_profit_price:
             self.logger.info("No valid Take Profit price provided, skipping TP order.")
             tp_success = True

        return sl_success


    async def _close_position(self, side: str, quantity: float, reason: str = "unknown", is_emergency: bool = False):
        """Places market order to close position, cancels related orders, updates state."""
        if side not in ['buy', 'sell'] or not quantity or quantity <= 0:
            self.logger.error(f"Invalid side '{side}' or quantity ({quantity}) for closing.")
            return

        log_prefix = "[EMERGENCY CLOSE]" if is_emergency else "[Closing Position]"
        self.logger.info(f"{log_prefix} Reason: {reason}. Attempting {side.upper()} market order for {quantity:.8f} {self.symbol}.")

        position_at_close = copy.deepcopy(self.current_position) if self.current_position else None
        if not position_at_close:
             self.logger.warning(f"{log_prefix} No current position found in state to close. Refreshing...")
             await self._refresh_position_and_orders()
             position_at_close = copy.deepcopy(self.current_position) if self.current_position else None
             if not position_at_close:
                  self.logger.error(f"{log_prefix} Still no position found after refresh. Cannot close.")
                  return

        # Use state quantity if different from requested, log warning
        state_qty = position_at_close.get('quantity', 0.0)
        if abs(quantity - state_qty) > FLOAT_EPSILON * 10:
             self.logger.warning(f"{log_prefix} Quantity to close ({quantity:.8f}) differs from state ({state_qty:.8f}). Using state quantity.")
             quantity = state_qty
             if quantity <= 0:
                  self.logger.error(f"{log_prefix} Position state quantity invalid ({quantity}). Cannot close.")
                  return

        await self._cancel_open_orders() # Cancel SL/TP first
        close_order_result = None
        filled_exit_details = None # Initialize here

        try:
            self.logger.info(f"{log_prefix} Placing MARKET order to close: {side.upper()} {quantity:.8f} {self.symbol}")
            close_order_result = await self.exchange_adapter.place_market_order(
                symbol=self.symbol, side=side.upper(), quantity=quantity, reduce_only=True
            )
            if not close_order_result or close_order_result.get('status') in ['REJECTED', 'EXPIRED', 'CANCELED']:
                 if isinstance(close_order_result, dict) and close_order_result.get('code') == -2022:
                      self.logger.warning(f"{log_prefix} ReduceOnly order rejected. Position likely already closed.")
                 else:
                      raise OrderExecutionError(f"Market close order failed or rejected. Result: {close_order_result}")

            self.logger.info(f"{log_prefix} Market close order initiated. ID: {close_order_result.get('orderId')}")
            await asyncio.sleep(1.5)
            exit_order_id = close_order_result.get('orderId')
            filled_exit_details = await self.exchange_adapter.get_order_info(self.symbol, exit_order_id) if exit_order_id else None # Assignment

            if not filled_exit_details or filled_exit_details.get('status') != 'FILLED':
                 self.logger.warning(f"{log_prefix} Close order {exit_order_id} not confirmed FILLED. Status: {filled_exit_details.get('status') if filled_exit_details else 'N/A'}. PnL may be inaccurate.")
                 current_price_fallback = await self.exchange_adapter.get_latest_price(self.symbol)
                 avg_exit_price = current_price_fallback if current_price_fallback and current_price_fallback > 0 else position_at_close.get('entryPrice', 0.0) # Fallback to entry price if no market price
                 executed_qty_close = quantity
            else:
                 executed_qty_close = float(filled_exit_details.get('executedQty', 0.0))
                 avg_exit_price = float(filled_exit_details.get('avgPrice', 0.0))
                 # This line is now safe as filled_exit_details is guaranteed to be defined (though possibly None)
                 if avg_exit_price <= 0 and executed_qty_close > 0: avg_exit_price = float(filled_exit_details.get('cumQuote', 0.0)) / executed_qty_close
                 if executed_qty_close <= 0 or avg_exit_price <= 0: raise OrderExecutionError(f"Close order {exit_order_id} filled but invalid qty/price.")
                 self.logger.info(f"{log_prefix} Close order FILLED. Qty: {executed_qty_close:.8f}, Price: {avg_exit_price:.8f}")

            self._update_state_after_close(position_at_close, executed_qty_close, avg_exit_price, reason, exit_order_id)
            await asyncio.sleep(0.5)
            await self._refresh_position_and_orders()
            if self.current_position is None: self.logger.info(f"{log_prefix} Position confirmed closed.")
            else:
                 self.logger.error(f"{log_prefix} Position state still shows open after close confirmed FILLED! Manual check needed!")
                 await self._send_notification("CRITICAL: Position possibly not closed despite FILLED order. Manual check!", level="critical")

            await self._async_save_current_state() # Use async wrapper

        except OrderExecutionError as e:
             self.logger.error(f"{log_prefix} Failed to execute closing order: {e}", exc_info=True)
             await self._send_notification(f"ERROR: Failed to close position ({reason}): {e}", level="error")
        except Exception as e:
             self.logger.critical(f"{log_prefix} Unexpected critical error during close_position: {e}", exc_info=True)
             await self._send_notification(f"CRITICAL ERROR during close_position ({reason}): {e}", level="critical")


    async def _cancel_open_orders(self):
        """Cancels all open SL/TP orders associated with the current symbol."""
        await self._refresh_position_and_orders() # Refresh first
        if not self.open_orders:
            self.logger.debug("No open orders found in state to cancel.")
            return

        order_ids_to_cancel = [order.get('orderId') for order in self.open_orders if order.get('symbol') == self.symbol]
        if not order_ids_to_cancel:
            self.logger.info("No open orders found for this symbol to cancel.")
            return

        self.logger.info(f"Cancelling {len(order_ids_to_cancel)} open order(s): {order_ids_to_cancel}")
        try:
            cancel_results = await self.exchange_adapter.cancel_multiple_orders(self.symbol, order_ids_to_cancel)
            self.logger.info(f"Cancellation attempt results: {cancel_results}")
            failed_cancels = [res for res in cancel_results if isinstance(res, dict) and res.get('status') == 'FAILED']
            if failed_cancels:
                 self.logger.warning(f"Failed to cancel some orders: {failed_cancels}")
                 await self._send_notification(f"WARNING: Failed to cancel orders: {failed_cancels}", level="warning")
        except Exception as e:
            self.logger.error(f"Error during order cancellation: {e}", exc_info=True)
            await self._send_notification(f"ERROR: Failed to cancel open orders: {e}", level="error")
        finally:
            self.open_orders = [] # Clear local state regardless of API result
            self.logger.debug("Local open_orders state cleared.")


    def _update_state_after_close(self, closed_position_details: Dict, executed_qty_close: float, avg_exit_price: float, reason: str, exit_order_id: Optional[Union[str, int]]):
        """Updates internal capital and records trade history after a close."""
        try:
            entry_price = float(closed_position_details.get('entryPrice', 0.0))
            direction = closed_position_details.get('direction')
            entry_fee = float(closed_position_details.get('entryFee', 0.0))

            entry_time_raw = closed_position_details.get('entryTime')
            entry_time_dt: Optional[datetime] = None

            if isinstance(entry_time_raw, str):
                try:
                    entry_time_dt = pd.Timestamp(entry_time_raw).to_pydatetime()
                    if entry_time_dt.tzinfo is None or entry_time_dt.tzinfo.utcoffset(entry_time_dt) is None:
                        entry_time_dt = entry_time_dt.replace(tzinfo=timezone.utc)
                    else:
                        entry_time_dt = entry_time_dt.astimezone(timezone.utc)
                except Exception as e_ts:
                    self.logger.error(f"Could not parse entryTime string '{entry_time_raw}' to datetime: {e_ts}. Storing as None.")
                    entry_time_dt = None
            elif isinstance(entry_time_raw, datetime): # Already a datetime object
                entry_time_dt = entry_time_raw
                if entry_time_dt.tzinfo is None or entry_time_dt.tzinfo.utcoffset(entry_time_dt) is None:
                    entry_time_dt = entry_time_dt.replace(tzinfo=timezone.utc)
                else:
                    entry_time_dt = entry_time_dt.astimezone(timezone.utc)
            elif entry_time_raw is not None: # Try to parse if it's some other type pandas can handle (e.g., pd.Timestamp)
                try:
                    entry_time_dt = pd.Timestamp(entry_time_raw).to_pydatetime()
                    if entry_time_dt.tzinfo is None or entry_time_dt.tzinfo.utcoffset(entry_time_dt) is None:
                        entry_time_dt = entry_time_dt.replace(tzinfo=timezone.utc)
                    else:
                        entry_time_dt = entry_time_dt.astimezone(timezone.utc)
                except Exception as e_ts_other:
                    self.logger.warning(f"entryTime ('{entry_time_raw}') of type {type(entry_time_raw)} could not be converted to datetime: {e_ts_other}. Storing as None.")
                    entry_time_dt = None
            else: # entry_time_raw is None
                self.logger.warning(f"entryTime is None. Storing as None.")
                entry_time_dt = None


            current_utc_time_for_exit = datetime.now(timezone.utc)

            if not direction or entry_price <= FLOAT_EPSILON or executed_qty_close <= FLOAT_EPSILON or avg_exit_price <= FLOAT_EPSILON:
                 self.logger.error(f"Cannot calculate PnL: Invalid closed position or exit data. Dir: {direction}, EntryP: {entry_price}, ExecQty: {executed_qty_close}, ExitP: {avg_exit_price}")
                 return

            direction_int = 1 if direction == 'long' else -1
            price_diff = avg_exit_price - entry_price
            gross_pnl = price_diff * executed_qty_close * direction_int
            exit_value = executed_qty_close * avg_exit_price
            exit_fee = exit_value * self.trading_fee_rate
            net_pnl_close = gross_pnl - exit_fee

            capital_before_this_close_pnl = self.internal_capital
            self.internal_capital += net_pnl_close
            total_fees_for_trade = entry_fee + exit_fee

            self.logger.info(f"Trade Closed PnL: Gross={gross_pnl:.4f}, ExitFee={exit_fee:.4f} (EntryFee was {entry_fee:.4f}), NetForThisClose={net_pnl_close:.4f}")
            self.logger.info(f"Internal Capital Updated: {capital_before_this_close_pnl:.2f} -> {self.internal_capital:.2f} (Change: {net_pnl_close:.2f})")

            trade_log = {
                'entry_time': entry_time_dt,
                'exit_time': current_utc_time_for_exit,
                'symbol': self.symbol,
                'interval': self.interval,
                'direction': direction,
                'entryPrice': entry_price,
                'exitPrice': avg_exit_price,
                'quantity': executed_qty_close,
                'leverage': self.leverage,
                'grossPnl': round(gross_pnl, 4),
                'entryFee': round(entry_fee, 4),
                'exitFee': round(exit_fee, 4),
                'net_pnl': round(net_pnl_close, 4),
                'total_fees_for_trade': round(total_fees_for_trade, 4),
                'exit_reason': reason,
                'finalCapital': round(self.internal_capital, 2),
                'entryOrderId': str(closed_position_details.get('entryOrderId')) if closed_position_details.get('entryOrderId') is not None else None,
                'exitOrderId': str(exit_order_id) if exit_order_id is not None else None,
                'slOrderId': str(closed_position_details.get('slOrderId')) if closed_position_details.get('slOrderId') is not None else None,
                'tpOrderId': str(closed_position_details.get('tpOrderId')) if closed_position_details.get('tpOrderId') is not None else None,
            }
            self.trade_history_buffer.append(trade_log)
            self.logger.info(f"Trade logged to history buffer. Reason: {reason}")

            if self.current_position and self.current_position.get('quantity', 0) - executed_qty_close <= FLOAT_EPSILON:
                 self.current_position = None
            elif self.current_position:
                 self.current_position['quantity'] = max(0, self.current_position['quantity'] - executed_qty_close)
                 if self.current_position['quantity'] <= FLOAT_EPSILON:
                      self.current_position = None
        except Exception as e:
             self.logger.error(f"Error updating state after close: {e}", exc_info=True)
             asyncio.create_task(self._send_notification(f"CRITICAL ERROR: Failed to update state after close: {e}", level="critical"))


    async def _adjust_quantity_for_exchange(self, quantity: float) -> Optional[float]:
        """Adjusts quantity based on exchange precision and minimums."""
        if quantity is None or quantity <= FLOAT_EPSILON: return None
        try:
            adjusted_qty = self.exchange_adapter.adjust_quantity_precision(self.symbol, quantity)
            if not adjusted_qty or adjusted_qty <= FLOAT_EPSILON:
                 self.logger.warning(f"Quantity {quantity:.8f} adjusted to invalid value ({adjusted_qty}).")
                 return None

            min_qty = self.exchange_adapter.get_min_quantity(self.symbol)
            if min_qty is not None and adjusted_qty < min_qty - FLOAT_EPSILON: # Use tolerance
                 self.logger.warning(f"Adjusted quantity {adjusted_qty:.8f} below min {min_qty:.8f}.")
                 return None

            min_notional = self.exchange_adapter.get_min_notional(self.symbol)
            if min_notional is not None:
                 current_price = await self.exchange_adapter.get_latest_price(self.symbol)
                 if current_price and current_price > 0:
                      notional_value = adjusted_qty * current_price
                      if notional_value < min_notional - FLOAT_EPSILON: # Use tolerance
                           self.logger.warning(f"Notional value {notional_value:.4f} below min {min_notional:.4f}.")
                           return None
                 else:
                      self.logger.warning("Could not get price for min notional check.")
                      return None
            return adjusted_qty
        except Exception as e:
            self.logger.error(f"Error adjusting quantity {quantity}: {e}", exc_info=True)
            return None


    async def _adjust_price_for_exchange(self, price: float) -> Optional[float]:
        """Adjusts price based on exchange precision."""
        if price is None or price <= FLOAT_EPSILON: return None
        try:
            adjusted_price = self.exchange_adapter.adjust_price_precision(self.symbol, price)
            if not adjusted_price or adjusted_price <= FLOAT_EPSILON:
                 self.logger.warning(f"Price {price:.8f} adjusted to invalid value ({adjusted_price}).")
                 return None
            return adjusted_price
        except Exception as e:
            self.logger.error(f"Error adjusting price {price}: {e}", exc_info=True)
            return None


    async def _send_notification(self, message: str, level: str = "info"):
        """Helper method to send notifications if manager is available."""
        if self.notification_manager:
            try:
                await self.notification_manager.send_notification(message, level=level)
            except Exception as e:
                self.logger.error(f"Failed to send notification: {e}", exc_info=True)
        else:
            self.logger.debug(f"Notification Manager not available. Message: [{level.upper()}] {message}")


    async def _async_save_current_state(self):
        """Async wrapper for the synchronous _save_current_state method."""
        try:
            # Run the synchronous save method in the default executor
            await asyncio.to_thread(self._save_current_state)
            self.logger.debug("Async save state task completed.")
        except Exception as e:
            self.logger.error(f"Error in async save state task: {e}", exc_info=True)
            # Optionally send notification about save failure
            await self._send_notification(f"ERROR: Async save state task failed: {e}", level="error")


    async def shutdown(self):
        """Performs graceful shutdown procedures."""
        if not self.is_running and self.stop_event.is_set():
            self.logger.info("Shutdown already in progress or complete.")
            return

        self.logger.info("\n" + "="*40 + "\n--- Initiating Graceful Shutdown ---\n" + "="*40)
        self.is_running = False
        self.stop_event.set()
        await self._send_notification("Initiating graceful shutdown...", level="info")

        # --- Close Open Position (if any) ---
        if self.current_position:
            self.logger.warning("Open position found during shutdown. Attempting to close.")
            await self._send_notification("WARNING: Open position found during shutdown. Closing...", level="warning")
            close_side = 'sell' if self.current_position.get('direction') == 'long' else 'buy'
            quantity_to_close = self.current_position.get('quantity')
            if quantity_to_close and quantity_to_close > 0:
                 await self._close_position(side=close_side, quantity=quantity_to_close, reason='shutdown', is_emergency=True)
            else:
                 self.logger.error("Cannot close position during shutdown: Invalid quantity in state.")
                 await self._cancel_open_orders()
        else:
            self.logger.info("No open position found. Cancelling any remaining open orders...")
            await self._cancel_open_orders()

        # --- Save Final State ---
        # Run the synchronous save method one last time
        self.logger.info("Saving final bot state...")
        try:
            # Use the async wrapper for the final save as well
            await self._async_save_current_state()
        except Exception as e:
            self.logger.error(f"Error during final state save: {e}", exc_info=True)

        # --- Close Exchange Connection ---
        if self.exchange_adapter:
            self.logger.info("Closing exchange connection...")
            try:
                await self.exchange_adapter.close_connection()
                self.logger.info("Exchange connection closed.")
            except Exception as e:
                self.logger.error(f"Error closing exchange connection: {e}", exc_info=True)

        self.logger.info("\n" + "="*40 + "\n--- Trading Bot Shutdown Complete ---\n" + "="*40)
        await self._send_notification("Shutdown complete. Bot has stopped.", level="info")


# --- Script Entry Point ---
if __name__ == "__main__":
    logger.info("--- Starting Trading Bot Script ---")

    bot_configuration = STRATEGY_CONFIG
    # Notifier instance is initialized globally

    bot_instance = None
    try:
        bot_instance = TradingBot(bot_configuration, notifier_instance)
        asyncio.run(bot_instance.run())

    except (ValueError, ConnectionError, RuntimeError, ExchangeConnectionError, TemporalSafetyError, FileNotFoundError) as e:
        logger.critical(f"Bot failed during initialization or setup: {e}", exc_info=True)
        if notifier_instance:
             logger.critical(f"CRITICAL STARTUP FAILURE NOTIFICATION (sync log): {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. asyncio.run() will handle graceful shutdown.")
        # The finally block in bot_instance.run() should trigger shutdown.
    except Exception as e:
        logger.critical(f"Unexpected critical error during bot execution: {e}", exc_info=True)
        if notifier_instance:
             logger.critical(f"CRITICAL UNEXPECTED ERROR NOTIFICATION (sync log): {e}")
        if bot_instance:
             logger.info("Attempting emergency shutdown due0 to unexpected error...")
             try:
                  asyncio.run(bot_instance.shutdown())
             except Exception as shutdown_err:
                  logger.error(f"Error during emergency shutdown: {shutdown_err}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("--- Trading Bot Script Finished ---")
