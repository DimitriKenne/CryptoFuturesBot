import pandas as pd
import numpy as np
import math
import logging
import time
import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timezone

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Define FLOAT_EPSILON for robust floating-point comparisons
FLOAT_EPSILON = 1e-9

# --- Technical Analysis Library Import ---
# Attempt to import 'ta' library for indicators. Handle missing library gracefully.
try:
    from ta.volatility import AverageTrueRange
    from ta.trend import EMAIndicator
    TA_AVAILABLE = True
except ImportError:
    # Log warning if 'ta' is not installed; fallback calculations will be attempted if needed.
    logging.warning(
        "Technical Analysis library 'ta' not found. Install using 'pip install ta'. "
        "Fallback ATR/EMA calculation will be attempted if features are missing."
    )
    AverageTrueRange = None
    EMAIndicator = None
    TA_AVAILABLE = False

# --- Configuration Imports ---
# Load configurations from the central config files. Assumes standard project structure.
try:
    from config.paths import PATHS
    from config.params import STRATEGY_CONFIG, BACKTESTER_CONFIG, EXCHANGE_CONFIG, FEATURE_CONFIG # Import FEATURE_CONFIG to know the volatility regime column name
except ImportError as e:
    logging.critical(f"Failed to import configuration modules (config/paths.py or config/params.py): {e}. Ensure files exist and structure is correct.", exc_info=True)
    raise # Stop execution if essential configs are missing
except Exception as e:
     logging.critical(f"Unexpected error importing configuration: {e}", exc_info=True)
     raise

# --- Logger Setup ---
# Get logger instance for this module. Assumes logging is configured externally.
logger = logging.getLogger(__name__)


class Backtester:
    """
    Simulates futures trading strategies based on model signals using historical data.

    Attributes:
        data (pd.DataFrame): OHLCV data with features.
        model_predict (pd.Series): Model prediction signals (-1, 0, 1).
        model_proba (pd.DataFrame): Model probability scores (columns: -1, 0, 1).
        symbol (str): Trading pair symbol (e.g., "BTCUSDT").
        interval (str): Data interval (e.g., "1h").
        model_type (str): Identifier for the model used.
        config (Dict[str, Any]): Combined configuration (Strategy + Backtester + Exchange).
        paths (Dict[str, Union[str, Path]]): Paths configuration.
        trades (List[Dict[str, Any]]): Log of executed trades.
        equity_curve (pd.Series): Time series of portfolio equity.
        current_balance (float): Current simulated account balance.
        current_equity (float): Current simulated account equity (balance + unrealized PnL).
        position_direction (int): Current position state (1: long, -1: short, 0: flat).
        entry_price (float): Entry price of the current open trade.
        entry_time (pd.Timestamp): Entry time of the current open trade.
        position_value_entry_usd (float): Notional value at entry.
        position_asset_qty (float): Size in base asset.
        liquidation_price (float): Estimated liquidation price for current trade.
        current_trade_sl_price (float): Stop loss price for the current trade.
        current_trade_tp_price (float): Take profit price for the current trade.
        trade_open_bar_index (int): iloc index of the bar where trade was entered.
        trade_max_holding_bars (Optional[int]): Max holding period for the current trade.
        exit_on_neutral_signal (bool): Config flag to control neutral signal exits.
        allow_long_trades (bool): Config flag to allow/block long entries.
        allow_short_trades (bool): Config flag to allow/block short entries.
        # Confidence filter parameters from config:
        confidence_filter_enabled (bool): Enable filtering trades based on model confidence?
        confidence_threshold_long (float): Minimum probability for a LONG signal (0.0-1.0).
        confidence_threshold_short (float): Minimum probability for a SHORT signal (0.0-1.0).
        # Volatility Regime Filter parameters from config:
        volatility_regime_filter_enabled (bool): Enable volatility regime filter?
        volatility_regime_max_holding_bars (Dict[int, Optional[int]]): Max holding per regime.
        allow_trading_in_volatility_regime (Dict[int, bool]): Allow trading per regime.
        # ... other state variables ...
    """

    def __init__(self,
                 data: pd.DataFrame,
                 model_predict: pd.Series,
                 model_proba: pd.DataFrame,
                 symbol: str,
                 interval: str,
                 model_type: str,
                 backtester_config_override: Optional[Dict[str, Any]] = None,
                 strategy_config_override: Optional[Dict[str, Any]] = None,
                 paths_override: Optional[Dict[str, Union[str, Path]]] = None):
        """
        Initializes the Backtester instance.

        Args:
            data (pd.DataFrame): DataFrame with OHLCV data and features, indexed by time.
                                 Must include 'open', 'high', 'low', 'close'.
            model_predict (pd.Series): Series with model predictions (-1, 0, 1) aligned with data index.
            model_proba (pd.DataFrame): DataFrame with model probability scores (columns: -1, 0, 1) aligned with data index.
            symbol (str): Trading pair symbol (e.g., "BTCUSDT").
            interval (str): Data interval (e.g., "1h").
            model_type (str): Type of ML model used (e.g., "lstm", "xgboost").
            backtester_config_override (Optional[Dict]): Overrides default BACKTESTER_CONFIG.
            strategy_config_override (Optional[Dict]): Overrides default STRATEGY_CONFIG.
            paths_override (Optional[Dict]): Overrides default PATHS config.
        """
        # FIX: Assign the module-level logger to an instance attribute
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"--- Initializing Backtester for {symbol} {interval} ({model_type}) ---")

        # Ensure data, model_predict, and model_proba are copies to prevent external modification
        self.data = data.copy()
        self.model_predict = model_predict.copy()
        self.model_proba = model_proba.copy() # Store the probability DataFrame

        self.symbol = symbol.upper()
        self.interval = interval
        self.model_type = model_type

        # --- Load and Combine Configurations ---
        # Start with deep copies of defaults, then overlay overrides and exchange specifics.
        try:
            strategy_conf = copy.deepcopy(STRATEGY_CONFIG)
            backtester_conf = copy.deepcopy(BACKTESTER_CONFIG)
            paths_conf = copy.deepcopy(PATHS) # Deep copy paths as well

            # Apply overrides if provided
            if strategy_config_override: strategy_conf.update(strategy_config_override)
            if backtester_config_override: backtester_conf.update(backtester_config_override)
            if paths_override: paths_conf.update(paths_override) # Simple update for paths

            # Determine exchange type (defaults to 'binance_futures' if not specified)
            self.exchange_type = strategy_conf.get('exchange_type', 'binance_futures')
            exchange_conf = copy.deepcopy(EXCHANGE_CONFIG) # Deep copy exchange config
            exchange_specific_params = exchange_conf.get(self.exchange_type, {})
            if not exchange_specific_params:
                 self.logger.warning(f"Configuration for exchange type '{self.exchange_type}' not found in EXCHANGE_CONFIG. Using defaults.")

            # Combine configurations: Backtester overrides Strategy, Exchange overrides both
            # Use ** to merge dictionaries. Order matters for overrides.
            self.config: Dict[str, Any] = {**strategy_conf, **backtester_conf, **exchange_specific_params}
            self.paths = paths_conf # Store the final paths config

        except Exception as e:
             self.logger.critical(f"Error loading or combining configurations: {e}", exc_info=True)
             raise ValueError("Configuration loading failed.") from e

        # --- Load and Validate Final Parameters ---
        # Populates instance attributes (e.e.g., self.initial_capital) from self.config
        try:
            self._load_and_validate_final_config()
        except ValueError as e:
             self.logger.critical(f"Configuration validation failed: {e}", exc_info=True)
             raise # Re-raise validation errors

        # --- Initialize Internal State ---
        self._initialize_internal_state()

        # --- Prepare Data ---
        # Validates columns, ensures types, calculates missing indicators if needed,
        # aligns signals and probabilities, and handles NaNs.
        try:
             self._prepare_data()
        except (ValueError, ImportError, KeyError) as e:
             self.logger.critical(f"Data preparation failed: {e}", exc_info=True)
             raise # Re-raise critical data prep errors

        # --- Log Initial Setup ---
        self._log_initial_setup()
        self.logger.info("--- Backtester Initialization Complete ---")


    def _load_and_validate_final_config(self):
        """Loads and validates parameters from the combined self.config dictionary."""
        self.logger.debug("Loading and validating final configuration parameters...")

        # Use .get() with default values for robustness
        self.initial_capital = float(self.config.get('initial_capital', 1000.0))
        self.risk_per_trade_pct = float(self.config.get('risk_per_trade_pct', 1.0)) # Percentage (0-100)
        self.leverage = float(self.config.get('leverage', 1.0))
        self.trading_fee_rate = float(self.config.get('trading_fee_rate', 0.0004)) # Fraction (e.g., 0.0004 for Binance Futures taker)
        self.maintenance_margin_rate = float(self.config.get('maintenance_margin_rate', 0.004)) # Fraction
        # max_holding_period is now primarily determined by volatility regime, but keep as a default/fallback
        self.max_holding_period = self.config.get('max_holding_period_bars') # int or None
        # Use specific liquidation fee if provided, else default to trading fee
        self.liquidation_fee_rate = float(self.config.get('liquidation_fee_rate', self.trading_fee_rate)) # Fraction
        self.max_concurrent_trades = int(self.config.get('max_concurrent_trades', 1))
        self.tie_breaker = str(self.config.get('tie_breaker', 'neutral')).lower()
        # New parameters for filtering entries
        self.exit_on_neutral_signal = bool(self.config.get('exit_on_neutral_signal', True))
        self.allow_long_trades = bool(self.config.get('allow_long_trades', True))
        self.allow_short_trades = bool(self.config.get('allow_short_trades', True))

        # --- Confidence Filter Parameters (from combined config) ---
        self.confidence_filter_enabled = bool(self.config.get('confidence_filter_enabled', False))
        # Get percentage thresholds and convert to fractions (0.0-1.0)
        confidence_threshold_long_pct = float(self.config.get('confidence_threshold_long_pct', 0.0))
        confidence_threshold_short_pct = float(self.config.get('confidence_threshold_short_pct', 0.0))
        self.confidence_threshold_long = confidence_threshold_long_pct / 100.0
        self.confidence_threshold_short = confidence_threshold_short_pct / 100.0

        # --- Volatility Regime Filter Parameters (NEW) ---
        self.volatility_regime_filter_enabled = bool(self.config.get('volatility_regime_filter_enabled', False))
        self.volatility_regime_max_holding_bars = self.config.get('volatility_regime_max_holding_bars', {0: None, 1: None, 2: None})
        self.allow_trading_in_volatility_regime = self.config.get('allow_trading_in_volatility_regime', {0: True, 1: True, 2: True})
        # Determine the expected column name for volatility regime from FEATURE_CONFIG defaults
        # Assuming the FeatureEngineer uses the shortest ATR period for the regime calculation
        fe_atr_periods = FEATURE_CONFIG.get('atr_periods', [14]) # Get default ATR periods from FE config
        atr_period_for_regime = min(fe_atr_periods) if fe_atr_periods else 14 # Use min period or default
        self.volatility_regime_col_name = f'volatility_regime' # Feature Engineer names it simply 'volatility_regime'


        # --- Exchange Specific Parameters (from EXCHANGE_CONFIG defaults or overrides) ---
        # Ensure these keys exist or provide sensible defaults if they are critical
        self.price_precision = int(self.config.get('price_precision', 8))
        self.quantity_precision = int(self.config.get('quantity_precision', 8))
        self.min_quantity = float(self.config.get('min_quantity', 0.0)) # Min trade size in base asset
        self.min_notional = float(self.config.get('min_notional', 0.0)) # Min trade value in quote asset

        # --- Volatility Adjustment Parameters ---
        self.volatility_adjustment_enabled = bool(self.config.get('volatility_adjustment_enabled', False))
        self.volatility_window_bars = int(self.config.get('volatility_window_bars', 14))
        self.fixed_take_profit_pct = float(self.config.get('fixed_take_profit_pct', 1.0)) # Percentage (0-100)
        self.fixed_stop_loss_pct = float(self.config.get('fixed_stop_loss_pct', 0.5))     # Percentage (0-100)
        self.alpha_take_profit = float(self.config.get('alpha_take_profit', 1.0))
        self.alpha_stop_loss = float(self.config.get('alpha_stop_loss', 1.5))
        # Define expected ATR column name based on the volatility window
        self.atr_vol_adj_col_name = f'atr_{self.volatility_window_bars}'

        # --- Trend Filter Parameters ---
        self.trend_filter_enabled = bool(self.config.get('trend_filter_enabled', False))
        self.trend_filter_ema_period = int(self.config.get('trend_filter_ema_period', 200))
        # Define expected EMA column name based on the filter period
        self.ema_filter_col_name = f'ema_{self.trend_filter_ema_period}'

        # --- Reporting Flags ---
        self.save_trades = bool(self.config.get('save_trades', True))
        self.save_equity_curve = bool(self.config.get('save_equity_curve', True))
        self.save_metrics = bool(self.config.get('save_metrics', True))

        # --- Convert Percentages to Fractions for Calculations ---
        self.risk_per_trade_fraction = self.risk_per_trade_pct / 100.0
        self.fixed_take_profit_fraction = self.fixed_take_profit_pct / 100.0
        self.fixed_stop_loss_fraction = self.fixed_stop_loss_pct / 100.0
        # FIX: Convert slippage tolerance percentage to fraction and store
        self.slippage_tolerance_fraction = float(self.config.get('slippage_tolerance_pct', 0.01)) # Load percentage first
        self.slippage_tolerance_fraction = self.slippage_tolerance_fraction / 100.0 # Convert to fraction


        # --- Validation Checks ---
        if self.initial_capital <= 0: raise ValueError("'initial_capital' must be positive.")
        if not (0 < self.risk_per_trade_fraction <= 1): raise ValueError("'risk_per_trade_pct' must be between 0 (exclusive) and 100 (inclusive).")
        if self.leverage <= 0: raise ValueError("'leverage' must be positive.")
        if not (0 <= self.trading_fee_rate < 1): raise ValueError("'trading_fee_rate' must be a non-negative fraction less than 1.")
        if not (0 <= self.maintenance_margin_rate < 1): raise ValueError("'maintenance_margin_rate' must be a non-negative fraction less than 1.")
        # Validate default max_holding_period if it's not None
        if self.max_holding_period is not None and self.max_holding_period <= 0: raise ValueError("'max_holding_period_bars' must be a positive integer or None.")
        if not (0 <= self.liquidation_fee_rate < 1): raise ValueError("'liquidation_fee_rate' must be a non-negative fraction less than 1.")
        if self.max_concurrent_trades != 1:
            self.logger.warning("Backtester currently only supports max_concurrent_trades=1. Forcing to 1.")
            self.max_concurrent_trades = 1
        if self.price_precision < 0: raise ValueError("'price_precision' must be non-negative.")
        if self.quantity_precision < 0: raise ValueError("'quantity_precision' must be non-negative.")
        if self.min_quantity < 0: raise ValueError("'min_quantity' must be non-negative.")
        if self.min_notional < 0: raise ValueError("'min_notional' must be non-negative.")
        if self.tie_breaker not in ['long', 'short', 'neutral']: raise ValueError("'tie_breaker' must be 'long', 'short', or 'neutral'.")
        if not isinstance(self.exit_on_neutral_signal, bool): raise ValueError("'exit_on_neutral_signal' must be a boolean.")
        # Validate new parameters
        if not isinstance(self.allow_long_trades, bool): raise ValueError("'allow_long_trades' must be a boolean.")
        if not isinstance(self.allow_short_trades, bool): raise ValueError("'allow_short_trades' must be a boolean.")
        if not self.allow_long_trades and not self.allow_short_trades:
             self.logger.warning("Both 'allow_long_trades' and 'allow_short_trades' are False. No trades will be taken.")
        # Validate confidence thresholds (now as fractions)
        if not (0.0 <= self.confidence_threshold_long <= 1.0):
             raise ValueError("'confidence_threshold_long_pct' must result in a fraction between 0.0 and 1.0.")
        if not (0.0 <= self.confidence_threshold_short <= 1.0):
             raise ValueError("'confidence_threshold_short_pct' must result in a fraction between 0.0 and 1.0.")
        # Validate slippage tolerance fraction
        if not (0.0 <= self.slippage_tolerance_fraction < 1.0):
             raise ValueError("'slippage_tolerance_pct' must result in a fraction between 0.0 and 1.0 (exclusive of 1.0).")

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


        # Volatility Adjustment specific validations
        if self.volatility_adjustment_enabled:
            if self.volatility_window_bars <= 0: raise ValueError("'volatility_window_bars' must be positive.")
            if self.alpha_take_profit < 0: raise ValueError("'alpha_take_profit' must be non-negative.")
            if self.alpha_stop_loss < 0: raise ValueError("'alpha_stop_loss' must be non-negative.")
        # Fixed TP/SL must always be positive (using fractions for check)
        if self.fixed_take_profit_fraction <= FLOAT_EPSILON: raise ValueError("'fixed_take_profit_pct' must be positive.")
        if self.fixed_stop_loss_fraction <= FLOAT_EPSILON: raise ValueError("'fixed_stop_loss_pct' must be positive.")

        # EMA Filter specific validations
        if self.trend_filter_enabled:
            if self.trend_filter_ema_period <= 0: raise ValueError("'trend_filter_ema_period' must be positive.")

        self.logger.debug("Configuration validation successful.")


    def _initialize_internal_state(self):
        """Initializes the backtester's internal state variables."""
        self.trades: List[Dict[str, Any]] = []
        # Initialize equity curve with the data's index and initial capital
        # This ensures the equity curve has the correct length and index from the start
        if not self.data.empty:
             self.equity_curve = pd.Series(index=self.data.index, dtype=float)
             self.equity_curve.iloc[0] = self.initial_capital
        else:
             # Handle empty data case gracefully
             self.equity_curve = pd.Series(dtype=float)
             self.logger.warning("Data is empty, equity curve initialized as empty.")

        self.current_balance = self.initial_capital
        self.current_equity = self.initial_capital # Starts same as balance
        self.position_direction = 0 # 1: long, -1: short, 0: flat
        self.entry_price = np.nan
        self.entry_time = pd.NaT # Use pandas NaT for datetime
        self.position_value_entry_usd = 0.0 # Notional value at entry
        self.position_asset_qty = 0.0      # Size in base asset
        self.liquidation_price = np.nan    # Estimated liquidation price for current trade
        self.current_trade_sl_price = np.nan # Stop loss price for the current trade
        self.current_trade_tp_price = np.nan # Take profit price for the current trade
        self.trade_open_bar_index = -1     # iloc index of the bar where trade was entered
        self.trade_max_holding_bars = None # NEW: Max holding period for the current trade


    def _prepare_data(self):
        """
        Validates required columns, ensures numeric types, calculates missing
        indicators (ATR/EMA) if needed/enabled, aligns signals and probabilities,
        and handles NaNs.
        """
        self.logger.debug("Preparing input data...")
        required_ohlc = ['open', 'high', 'low', 'close']
        if not all(col in self.data.columns for col in required_ohlc):
            missing = [col for col in required_ohlc if col not in self.data.columns]
            raise ValueError(f"Input data missing required OHLC columns: {missing}")

        # Ensure OHLC columns are numeric
        for col in required_ohlc:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Check for NaNs in OHLC after coercion
        if self.data[required_ohlc].isnull().any().any():
            self.logger.warning("NaN values found in OHLC data after conversion. These rows might be dropped or cause issues.")
            # Option: Impute or drop NaNs in OHLC here if not done in preprocessing
            pass


        # --- Calculate Missing ATR if Volatility Adjustment Enabled ---
        if self.volatility_adjustment_enabled and self.atr_vol_adj_col_name not in self.data.columns:
            self._calculate_missing_indicator('ATR', self.atr_vol_adj_col_name, self.volatility_window_bars)

        # --- Calculate Missing EMA if Trend Filter Enabled ---
        if self.trend_filter_enabled and self.ema_filter_col_name not in self.data.columns:
            self._calculate_missing_indicator('EMA', self.ema_filter_col_name, self.trend_filter_ema_period)

        # Ensure calculated indicator columns are numeric
        for col_name in [self.atr_vol_adj_col_name, self.ema_filter_col_name]:
             if col_name in self.data.columns:
                  self.data[col_name] = pd.to_numeric(self.data[col_name], errors='coerce')
                  # Fill NaNs for indicators if they were just calculated (often first 'window' bars are NaN)
                  if self.data[col_name].isnull().any():
                       self.logger.warning(f"NaN values found in calculated indicator '{col_name}'. Forward filling NaNs.")
                       self.data[col_name].ffill(inplace=True)
                       # If there are still NaNs at the beginning, fill with a default or drop
                       if self.data[col_name].isnull().any():
                            self.logger.warning(f"NaN values still present in '{col_name}' after ffill. Filling remaining with mean/median or dropping rows.")
                            # Decide on strategy: drop rows, fill with mean/median, or fill with a constant
                            # For simplicity, let's drop rows that still have NaN in critical indicator columns after ffill
                            pass # Handled in final NaN drop below


        # Ensure Volatility Regime column exists and is of nullable integer type (Int8Dtype)
        # FeatureEngineer is expected to create this column. Check here defensively.
        if self.volatility_regime_filter_enabled:
             if self.volatility_regime_col_name not in self.data.columns:
                  self.logger.error(f"Volatility regime filter enabled, but column '{self.volatility_regime_col_name}' is missing from data.")
                  # Add a column of NaNs to prevent errors, but log critical failure
                  self.data[self.volatility_regime_col_name] = pd.NA
                  self.logger.critical("Volatility regime filter cannot operate without the required feature column.")
                  # Consider disabling the filter here if the column is essential
                  # self.volatility_regime_filter_enabled = False
             else:
                  # Ensure the column is of the correct type (nullable integer)
                  if not isinstance(self.data[self.volatility_regime_col_name].dtype, pd.Int8Dtype):
                       self.logger.warning(f"Volatility regime column '{self.volatility_regime_col_name}' has incorrect dtype {self.data[self.volatility_regime_col_name].dtype}. Attempting cast to Int8Dtype.")
                       try:
                            self.data[self.volatility_regime_col_name] = self.data[self.volatility_regime_col_name].astype(pd.Int8Dtype())
                       except Exception as e:
                            self.logger.error(f"Failed to cast '{self.volatility_regime_col_name}' to Int8Dtype: {e}. NaNs in this column may cause errors.", exc_info=True)


        # --- Align Model Predictions and Probabilities ---
        # Reindex predictions to match data index, fill missing with 0 (neutral), ensure int type
        # This handles cases where predictions might not cover the full data range (e.g., LSTM sequences)
        if not self.model_predict.index.equals(self.data.index):
             self.logger.warning("Model prediction index does not match data index. Reindexing predictions.")
             self.data['signal'] = self.model_predict.reindex(self.data.index).fillna(0).astype(int)
        else:
             self.data['signal'] = self.model_predict.fillna(0).astype(int) # Just fill NaNs if index matches

        self.logger.debug("Model predictions aligned with data index.")

        # Align probabilities to match data index, fill missing with NaN
        if not self.model_proba.index.equals(self.data.index):
             self.logger.warning("Model probability index does not match data index. Reindexing probabilities.")
             # Reindex to data index, fill missing rows with NaN
             self.model_proba = self.model_proba.reindex(self.data.index)
        # Ensure expected columns (-1, 0, 1) exist after reindexing, fill missing with NaN
        expected_proba_cols = [-1, 0, 1]
        for col in expected_proba_cols:
             if col not in self.model_proba.columns:
                  self.model_proba[col] = np.nan
                  self.logger.warning(f"Probability DataFrame missing column {col}. Added with NaN values.")
        # Ensure probability columns are numeric
        for col in expected_proba_cols:
             if col in self.model_proba.columns:
                  self.model_proba[col] = pd.to_numeric(self.model_proba[col], errors='coerce')


        self.logger.debug("Model probabilities aligned with data index.")


        # --- Final Data Cleaning (Drop NaNs in Critical Columns) ---
        # Define columns essential for the backtest loop to function
        # Include probability columns for the predicted signal if confidence threshold is used
        # Include volatility regime column if filter is enabled
        critical_cols = required_ohlc + ['signal']
        if self.volatility_adjustment_enabled:
            critical_cols.append(self.atr_vol_adj_col_name)
        if self.trend_filter_enabled:
            critical_cols.append(self.ema_filter_col_name)
        # Add the volatility regime column to critical columns if the filter is enabled
        if self.volatility_regime_filter_enabled:
             critical_cols.append(self.volatility_regime_col_name)
        # Add the probability columns if confidence filtering is enabled
        if self.confidence_filter_enabled:
             # We need the probability for the *specific* predicted signal (-1 or 1)
             # The check for NaN in the required probability will be done within _apply_entry_filters.
             # So, we don't add the specific probability column to critical_cols here,
             # but we rely on _apply_entry_filters to handle the NaN case.
             pass # No change needed here for probability NaN check


        # Ensure all critical columns actually exist in the data before dropping NaNs
        critical_cols_present = [col for col in critical_cols if col in self.data.columns]

        initial_rows = len(self.data)
        # FIX: Removed repeated 'inplace=True' argument
        self.data.dropna(subset=critical_cols_present, inplace=True)
        # Ensure data is a standalone copy after dropping NaNs
        self.data = self.data.copy()
        rows_removed = initial_rows - len(self.data)
        if rows_removed > 0:
            self.logger.warning(f"Removed {rows_removed} rows with NaNs in critical columns ({critical_cols_present}) after data preparation.")

        if self.data.empty:
            raise ValueError("DataFrame is empty after removing NaNs in critical columns. Cannot run backtest.")

        # After dropping rows from self.data, realign model_predict and model_proba again
        # to ensure they have the same index as the cleaned self.data
        if not self.model_predict.index.equals(self.data.index):
             self.logger.warning("Realigning model predictions after final data cleaning.")
             self.data['signal'] = self.model_predict.reindex(self.data.index).fillna(0).astype(int)
        if not self.model_proba.index.equals(self.data.index):
             self.logger.warning("Realigning model probabilities after final data cleaning.")
             self.model_proba = self.model_proba.reindex(self.data.index)
             # Fill any newly introduced NaNs in probability columns after reindexing
             expected_proba_cols = [-1, 0, 1] # Redefine locally if needed
             self.model_proba[expected_proba_cols] = self.model_proba[expected_proba_cols].fillna(np.nan)


        self.logger.debug("Data preparation complete.")


    def _calculate_missing_indicator(self, indicator_type: str, col_name: str, window: int):
        """Calculates missing ATR or EMA using the 'ta' library if available."""
        self.logger.warning(f"Required {indicator_type} column '{col_name}' not found. Attempting fallback calculation...")
        if not TA_AVAILABLE:
            raise ImportError(f"Cannot calculate {indicator_type}: 'ta' library not installed or import failed.")

        try:
            if indicator_type == 'ATR' and AverageTrueRange:
                # Ensure OHLC columns are present and numeric before calculating
                required_ohlc = ['open', 'high', 'low', 'close']
                if not all(col in self.data.columns and pd.api.types.is_numeric_dtype(self.data[col]) for col in required_ohlc):
                     self.logger.error(f"Cannot calculate ATR: Missing or non-numeric OHLC data.")
                     raise ValueError("Invalid OHLC data for ATR calculation.")

                indicator = AverageTrueRange(
                    high=self.data['high'], low=self.data['low'], close=self.data['close'],
                    window=window, fillna=False # Calculate without filling NaNs initially
                )
                self.data[col_name] = indicator.average_true_range()
                self.logger.info(f"Successfully calculated and added missing {indicator_type} column '{col_name}'.")

            elif indicator_type == 'EMA' and EMAIndicator:
                 # Ensure close column is present and numeric before calculating
                 if 'close' not in self.data.columns or not pd.api.types.is_numeric_dtype(self.data['close']):
                      self.logger.error(f"Cannot calculate EMA: Missing or non-numeric close data.")
                      raise ValueError("Invalid close data for EMA calculation.")

                 indicator = EMAIndicator(
                    close=self.data['close'], window=window, fillna=False # Calculate without filling NaNs initially
                 )
                 self.data[col_name] = indicator.ema_indicator()
                 self.logger.info(f"Successfully calculated and added missing {indicator_type} column '{col_name}'.")

            else:
                # This case should ideally not be reached if indicator_type is 'ATR' or 'EMA'
                # but serves as a safeguard.
                raise NotImplementedError(f"Fallback calculation for {indicator_type} not implemented or library component missing.")

        except Exception as e:
            self.logger.error(f"Failed to calculate fallback {indicator_type} '{col_name}': {e}", exc_info=True)
            # Add NaN column to prevent key errors later, but log the failure
            self.data[col_name] = np.nan
            self.logger.error(f"{indicator_type}-based functionality cannot operate without valid data. Consider disabling or ensuring feature is generated.")
            # Option: Disable the feature if calculation fails, e.g.,
            # if indicator_type == 'ATR': self.volatility_adjustment_enabled = False
            # if indicator_type == 'EMA': self.trend_filter_enabled = False


    def _log_initial_setup(self):
        """Logs the key configuration parameters after initialization."""
        self.logger.info(f"Initial Capital: {self.initial_capital:.2f}, Leverage: {self.leverage}x")
        self.logger.info(f"Risk/Trade: {self.risk_per_trade_pct:.2f}%, Fee Rate: {self.trading_fee_rate:.5f}")
        # Log default max holding, but note it can be overridden by regime
        self.logger.info(f"Default Max Holding: {self.max_holding_period} bars" if self.max_holding_period else "Default Max Holding: None")
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


        if self.volatility_adjustment_enabled:
            self.logger.info(f"Volatility Adjustment: ENABLED (Window={self.volatility_window_bars}, Fixed TP={self.fixed_take_profit_pct:.2f}%, Fixed SL={self.fixed_stop_loss_pct:.2f}%, Alpha TP={self.alpha_take_profit}, Alpha SL={self.alpha_stop_loss})")
        else:
            self.logger.info(f"Volatility Adjustment: DISABLED (Using Fixed TP={self.fixed_take_profit_pct:.2f}%, Fixed SL={self.fixed_stop_loss_pct:.2f}%)")

        if self.trend_filter_enabled:
            self.logger.info(f"Trend Filter: ENABLED (EMA Period={self.trend_filter_ema_period})")
        else:
            self.logger.info("Trend Filter: DISABLED")

        self.logger.info(f"Exchange Config: Price Precision={self.price_precision}, Qty Precision={self.quantity_precision}, Min Qty={self.min_quantity}, Min Notional={self.min_notional}")
        self.logger.info(f"Slippage Tolerance: {self.slippage_tolerance_fraction*100:.2f}%") # Log the correctly stored fraction


    def run_backtest(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Runs the backtest simulation loop over the prepared data.

        Returns:
            Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
                - trades_df: DataFrame of executed trades.
                - equity_curve: Series representing equity over time.
                - summary_metrics: Dictionary of performance metrics.
        """
        self.logger.info(f"--- Starting Backtest Simulation ({len(self.data)} bars) ---")
        start_time = time.time()

        if self.data.empty:
            self.logger.error("Cannot run backtest: Data is empty after preparation.")
            # Return empty results but with initial capital in metrics
            metrics_on_empty = self._calculate_summary_metrics(pd.DataFrame()) # Pass empty df
            return pd.DataFrame(), pd.Series(dtype=float), metrics_on_empty

        # Ensure equity curve is initialized with the correct index and initial capital
        # This was moved to __init__ but re-checked here defensively
        if self.equity_curve.empty or not self.equity_curve.index.equals(self.data.index):
             self.logger.warning("Equity curve not initialized correctly. Re-initializing.")
             self._initialize_internal_state() # Re-initialize all state if equity curve is wrong

        # --- Main Simulation Loop ---
        # Iterate through each bar index (iloc)
        # Entry happens on open of bar i+1 based on signal at bar i
        # Exit checks happen based on prices within bar i
        for i in range(len(self.data)):
            current_bar = self.data.iloc[i]
            current_timestamp = self.data.index[i]
            current_signal = current_bar['signal']
            is_last_bar = (i == len(self.data) - 1)

            self.logger.debug(f"Processing bar {i} ({current_timestamp}), Signal: {current_signal}, Position: {self.position_direction}")

            # --- 1. Process Exits for Existing Positions ---
            # Check for exits *before* checking entries for the next bar
            if self.position_direction != 0:
                # Pass the current bar index to check_exit_conditions
                exit_trade, exit_reason, exit_price = self._check_exit_conditions(i)
                if exit_trade:
                    # If an exit condition (SL, TP, Liq, Time Limit, Invalid OHLC) is met within bar i,
                    # close the position at the determined exit_price.
                    self._close_position(i, exit_reason, exit_price)
                    # Equity/Balance updated in _close_position.
                    # After closing, we are flat. The loop continues to the next bar,
                    # where a new entry might be considered based on the signal from bar i.
                else:
                    # If no hard exit condition is met, check for signal-based exit/reversal at the *close* of bar i.
                    # This happens if the signal changes direction or becomes neutral while in a position.
                    # Only check signal exit/reversal if *not* already exited by SL/TP/etc.

                    # --- Reversal Logic ---
                    # If currently Long and signal is Short, OR currently Short and signal is Long
                    if (self.position_direction == 1 and current_signal == -1) or \
                       (self.position_direction == -1 and current_signal == 1):

                        # Determine the side for the potential new position
                        new_position_side = 'buy' if current_signal == 1 else 'sell'

                        # Check if trades in the new direction are allowed
                        if (current_signal == 1 and self.allow_long_trades) or \
                           (current_signal == -1 and self.allow_short_trades):

                            # Apply entry filters to the *opposite* signal using data from the current bar (i)
                            # Pass the current bar data and probabilities
                            current_probabilities = self.model_proba.loc[current_timestamp] # Get probabilities for current bar
                            # Pass the current bar data (iloc[i]) for filter checks
                            if self._apply_entry_filters(i, current_signal): # Filters applied based on bar i data and signal

                                self.logger.info(f"Index {i} ({current_timestamp}): Reversal signal ({current_signal}) received and filters passed. Currently {('LONG' if self.position_direction == 1 else 'SHORT')}. Attempting reversal.")

                                # 1. Close the current position
                                # Exit at the close price of the current bar with slippage
                                exit_price_for_close = current_bar['close'] * (1 - self.slippage_tolerance_fraction if self.position_direction == 1 else 1 + self.slippage_tolerance_fraction)
                                self._close_position(i, 'signal_reverse', exit_price_for_close)

                                # 2. Immediately attempt to open the new position in the opposite direction
                                # Entry for the new position will be simulated on the *open* of the *next* bar (i+1)
                                # Pass the current bar index (i) and the reversal signal
                                self._execute_entry(i, current_signal)

                            else:
                                self.logger.debug(f"Index {i} ({current_timestamp}): Reversal signal ({current_signal}) received but filters blocked the new entry. Holding current position.")
                        else:
                            self.logger.debug(f"Index {i} ({current_timestamp}): Reversal signal ({current_signal}) received, but trades in the new direction ({new_position_side.upper()}) are disabled. Holding current position.")

                    # --- Neutral Signal Exit Logic (only if not reversing) ---
                    # Apply the new config parameter here: only exit on neutral if enabled AND not reversing
                    elif current_signal == 0 and self.exit_on_neutral_signal:
                        self.logger.debug(f"Index {i} ({current_timestamp}): Neutral signal detected and exit_on_neutral_signal is True. Exiting at close {current_bar['close']:.{self.price_precision}f}.")
                        # Exit at the close price of the current bar with slippage
                        exit_price_for_neutral = current_bar['close'] * (1 - self.slippage_tolerance_fraction if self.position_direction == 1 else 1 + self.slippage_tolerance_fraction)
                        self._close_position(i, 'signal_neutral', exit_price_for_neutral) # Use 'signal_neutral' as reason


            # --- 2. Process Entries for New Positions (if flat) ---
            # Entry attempted only if currently flat, signal is non-neutral, filters pass, and not the last bar
            # Entry happens on the *open* of the *next* bar (i+1) based on the signal from bar (i).
            # This block is also where the *second* part of a reversal (opening the new position)
            # would be handled if the close happened in the *previous* bar and left the bot flat.
            # However, with the immediate reversal logic added above, this section will primarily
            # handle initial entries when starting flat, or entries after a non-signal-based exit (SL/TP/Liq/Time).
            if self.position_direction == 0 and current_signal != 0 and not is_last_bar:
                 # Check if the next bar exists before attempting entry logic
                 if i + 1 < len(self.data):
                      # Add check for allowed trade direction here
                      if (current_signal == 1 and self.allow_long_trades) or \
                         (current_signal == -1 and self.allow_short_trades):
                           # Apply entry filters here, based on data from the current bar (i)
                           # Pass the current bar data and probabilities
                           # Pass the current bar data (iloc[i]) for filter checks
                           if self._apply_entry_filters(i, current_signal): # Filters applied based on bar i data and signal
                                # Pass the current bar index (i) and the signal for entry on bar i+1
                                self._execute_entry(i, current_signal)
                           else:
                                self.logger.debug(f"Index {i}: Signal {current_signal} blocked by entry filters.")
                      else:
                           self.logger.debug(f"Index {i}: Signal {current_signal} blocked by trade direction filter (allow_long_trades={self.allow_long_trades}, allow_short_trades={self.allow_short_trades}).")
                 else:
                      self.logger.debug(f"Index {i}: Last bar reached, skipping entry attempt for next bar.")


            # --- 3. Update Equity Curve (End of Bar i) ---
            # Calculates equity based on balance and unrealized PnL at the close of bar i
            # This is done for *every* bar, regardless of position state, to track equity fluctuations.
            self._update_equity_curve(i)

        # --- End of Loop ---

        # Final position closure if still open on the last bar
        if self.position_direction != 0:
             last_bar_index = len(self.data) - 1
             last_bar_close = self.data.iloc[last_bar_index]['close']
             self.logger.info(f"End of data reached. Closing open position at final close price {last_bar_close:.{self.price_precision}f}.")
             # Exit at the close price of the last bar with slippage
             exit_price_for_end = last_bar_close * (1 - self.slippage_tolerance_fraction if self.position_direction == 1 else 1 + self.slippage_tolerance_fraction)
             self._close_position(last_bar_index, 'end_of_data', exit_price_for_end)
             # Update equity one last time for the final bar after closing
             self._update_equity_curve(last_bar_index)


        # --- Finalize Results ---
        # Fill any potential NaNs in equity curve (e.e.g., from initial bars before first trade)
        # This should ideally be handled by initializing the equity curve with the full index,
        # but ffill/fillna provides robustness against unexpected gaps.
        self.equity_curve.ffill(inplace=True)
        self.equity_curve.fillna(self.initial_capital, inplace=True) # Fill any remaining NaNs with initial capital

        end_time = time.time()
        self.logger.info(f"--- Backtest Simulation Finished ({end_time - start_time:.2f} seconds) ---")

        # Convert trades list to DataFrame
        trades_df = pd.DataFrame(self.trades)

        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(trades_df)

        # Perform PnL consistency check
        self._check_pnl_consistency(summary_metrics)

        # Save results using the internal saving mechanism
        self._save_results(trades_df, self.equity_curve, summary_metrics)

        return trades_df, self.equity_curve, summary_metrics


    def _apply_entry_filters(self, signal_bar_index: int, signal: int) -> bool:
        """
        Applies configured filters (e.g., EMA trend, confidence threshold, volatility regime)
        to an entry signal. Filters are based on data from the signal_bar_index.
        """
        # Signal is guaranteed to be non-zero here (1 or -1)
        current_bar = self.data.iloc[signal_bar_index]
        current_timestamp = self.data.index[signal_bar_index]
        current_close = current_bar['close'] # Use close of the signal bar for filter check

        # --- Volatility Regime Filter (NEW) ---
        if self.volatility_regime_filter_enabled:
             try:
                  # Get the volatility regime for the current bar
                  current_regime = current_bar.get(self.volatility_regime_col_name)

                  # Check if the regime value is valid (0, 1, or 2) and not NaN
                  if pd.isna(current_regime) or current_regime not in [0, 1, 2]:
                       self.logger.warning(f"Index {signal_bar_index} ({current_timestamp}): Volatility regime value ({current_regime}) is invalid or NaN. Volatility regime filter blocks entry.")
                       return False # Block entry if regime is invalid

                  # Check if trading is allowed in this regime
                  if not self.allow_trading_in_volatility_regime.get(current_regime, False): # Default to False if regime key missing (shouldn't happen with validation)
                       self.logger.debug(f"Index {signal_bar_index} ({current_timestamp}): Trading is not allowed in volatility regime {current_regime}. Volatility regime filter blocks signal {signal}.")
                       return False # Block entry if trading is not allowed in this regime
                  self.logger.debug(f"Index {signal_bar_index} ({current_timestamp}): Trading is allowed in volatility regime {current_regime}. Passed volatility regime filter.")

             except KeyError:
                  self.logger.error(f"Volatility regime filter enabled, but column '{self.volatility_regime_col_name}' not found in data. Disabling filter and blocking trade.")
                  self.volatility_regime_filter_enabled = False # Disable filter permanently if column missing
                  return False # Block trade
             except Exception as e:
                  self.logger.error(f"Error during volatility regime filter check at index {signal_bar_index} ({current_timestamp}): {e}. Blocking entry.", exc_info=True)
                  return False # Block entry on filter error


        # --- Confidence Threshold Filter ---
        if self.confidence_filter_enabled:
             # Determine the correct threshold based on the signal direction
             confidence_threshold = self.confidence_threshold_long if signal == 1 else self.confidence_threshold_short

             # If the threshold is 0 or less, the filter is effectively disabled for this direction
             if confidence_threshold > FLOAT_EPSILON:
                  try:
                       # Get the probability for the predicted signal from the model_proba DataFrame
                       # Use .loc for index-based lookup
                       confidence = self.model_proba.loc[current_timestamp, signal]
                       if pd.isna(confidence):
                            self.logger.debug(f"Index {signal_bar_index} ({current_timestamp}): Confidence score is NaN for signal {signal}. Confidence filter blocks entry.")
                            return False # Block entry if confidence is NaN
                       if confidence < confidence_threshold:
                            self.logger.debug(f"Index {signal_bar_index} ({current_timestamp}): Confidence {confidence:.2f} for signal {signal} is below threshold {confidence_threshold:.2f}. Confidence filter blocks entry.")
                            return False # Block entry if confidence is below threshold
                       self.logger.debug(f"Index {signal_bar_index} ({current_timestamp}): Confidence {confidence:.2f} for signal {signal} meets threshold.")
                  except KeyError:
                       self.logger.warning(f"Index {signal_bar_index} ({current_timestamp}): Could not find probability for signal {signal} in model_proba DataFrame. Confidence filter blocks entry.")
                       return False # Block entry if probability column is missing
                  except Exception as e:
                       self.logger.error(f"Error during confidence filter check at index {signal_bar_index} ({current_timestamp}): {e}. Blocking entry.", exc_info=True)
                       return False # Block entry on filter error


        # --- EMA Trend Filter ---
        if self.trend_filter_enabled:
            try:
                ema_value = current_bar.get(self.ema_filter_col_name)
                if pd.isna(ema_value):
                    self.logger.warning(f"Index {signal_bar_index} ({current_timestamp}): EMA value is NaN. Trend filter blocks signal {signal}.")
                    return False # Block trade if EMA is NaN

                # Long only if close > EMA; Short only if close < EMA
                # Add a small tolerance (FLOAT_EPSILON) for floating point comparisons
                if (signal == 1 and current_close <= ema_value + FLOAT_EPSILON) or \
                   (signal == -1 and current_close >= ema_value - FLOAT_EPSILON):
                    self.logger.debug(f"Index {signal_bar_index}: EMA filter blocked signal {signal}. Close={current_close:.{self.price_precision}f}, EMA={ema_value:.{self.price_precision}f}")
                    return False
                self.logger.debug(f"Index {signal_bar_index}: Signal {signal} passed EMA trend filter.")

            except KeyError:
                self.logger.error(f"EMA filter column '{self.ema_filter_col_name}' not found. Disabling filter and blocking trade.")
                self.trend_filter_enabled = False # Disable filter permanently if column missing
                return False # Block trade
            except Exception as e:
                self.logger.error(f"Error during EMA filter check at index {signal_bar_index}: {e}. Blocking trade.", exc_info=True)
                return False # Block trade on filter error

        # --- Add other entry filters here ---
        # Example: Volatility filter (e.e.g., block if ATR is too low/high)

        return True # Signal passes all filters


    def _execute_entry(self, signal_bar_index: int, signal: int):
        """Handles the logic to enter a new trade on the open of the next bar."""
        entry_bar_index = signal_bar_index + 1
        # Ensure entry bar exists within the data (checked before calling this function, but defensive here)
        if entry_bar_index >= len(self.data):
            self.logger.debug(f"Index {signal_bar_index}: No next bar for entry. Skipping entry logic.")
            return

        entry_bar = self.data.iloc[entry_bar_index]
        entry_price = entry_bar['open']
        entry_timestamp = self.data.index[entry_bar_index]

        if pd.isna(entry_price) or entry_price <= FLOAT_EPSILON:
            self.logger.warning(f"Index {signal_bar_index}: Invalid entry price ({entry_price}) on next bar {entry_bar_index} ({entry_timestamp}). Skipping entry.")
            return

        # --- Set Position Direction ---
        self.position_direction = signal # 1 for long, -1 for short

        # --- Determine Max Holding Period for This Trade (NEW) ---
        self.trade_max_holding_bars = self.max_holding_period # Default to global config value
        if self.volatility_regime_filter_enabled:
             # Get the volatility regime for the signal bar (bar i)
             signal_bar = self.data.iloc[signal_bar_index]
             current_regime = signal_bar.get(self.volatility_regime_col_name)

             if pd.notna(current_regime) and current_regime in [0, 1, 2]:
                  regime_holding = self.volatility_regime_max_holding_bars.get(current_regime)
                  if regime_holding is not None and (isinstance(regime_holding, int) and regime_holding > 0):
                       self.trade_max_holding_bars = regime_holding
                       self.logger.debug(f"Index {signal_bar_index}: Setting trade max holding to {self.trade_max_holding_bars} based on volatility regime {current_regime}.")
                  elif regime_holding is None:
                       self.trade_max_holding_bars = None # No time limit for this regime
                       self.logger.debug(f"Index {signal_bar_index}: Setting trade max holding to None based on volatility regime {current_regime}.")
                  else:
                       self.logger.warning(f"Index {signal_bar_index}: Invalid max holding period ({regime_holding}) configured for regime {current_regime}. Using default ({self.max_holding_period}).")
             else:
                  self.logger.warning(f"Index {signal_bar_index}: Volatility regime ({current_regime}) invalid for determining trade max holding. Using default ({self.max_holding_period}).")


        # --- Calculate Dynamic Barriers ---
        # Use volatility from the *signal* bar for barrier calculation
        signal_bar_atr = self.data.iloc[signal_bar_index].get(self.atr_vol_adj_col_name) # Use .get() for safety
        calculated_tp_price, calculated_sl_price = self._calculate_dynamic_barriers(entry_price, signal_bar_atr)

        # Validate calculated SL price - SL is critical for risk management
        if pd.isna(calculated_sl_price) or calculated_sl_price <= FLOAT_EPSILON:
             self.logger.warning(f"Index {signal_bar_index}: Failed to calculate valid SL barrier for entry price {entry_price:.{self.price_precision}f} (SL={calculated_sl_price}). Skipping entry.")
             self._reset_position_state() # Reset direction if SL invalid
             return
        # Validate calculated TP price - TP can be NaN, but log if it's invalid when expected
        if pd.notna(calculated_tp_price) and calculated_tp_price <= FLOAT_EPSILON:
             self.logger.warning(f"Index {signal_bar_index}: Calculated TP barrier is invalid ({calculated_tp_price}). Proceeding without TP.")
             calculated_tp_price = np.nan # Set TP to NaN if invalid

        # --- Calculate Position Size ---
        position_asset_qty, position_value_entry_usd = self._calculate_position_size(entry_price, calculated_sl_price)

        if position_asset_qty <= FLOAT_EPSILON:
            self.logger.debug(f"Index {signal_bar_index}: Position size calculated as zero or less ({position_asset_qty:.{self.quantity_precision}f}). Skipping entry.")
            self._reset_position_state() # Reset direction if size is zero
            return

        # --- Calculate Margin and Fees ---
        initial_margin = position_value_entry_usd / self.leverage
        entry_fee = position_value_entry_usd * self.trading_fee_rate

        # --- Final Capital Check ---
        required_capital = initial_margin + entry_fee
        # Add a small tolerance (FLOAT_EPSILON) for floating point comparison
        if required_capital > self.current_balance + FLOAT_EPSILON:
             self.logger.warning(f"Index {signal_bar_index} ({entry_timestamp}): Insufficient balance ({self.current_balance:.2f}) for entry. Required: {required_capital:.2f} (Margin: {initial_margin:.2f}, Fee: {entry_fee:.4f}). Skipping entry.")
             self._reset_position_state()
             return

        # --- Finalize Entry State ---
        self.entry_price = entry_price
        self.entry_time = entry_timestamp
        self.position_asset_qty = position_asset_qty
        self.position_value_entry_usd = position_value_entry_usd
        self.current_trade_tp_price = calculated_tp_price # Store potentially NaN TP
        self.current_trade_sl_price = calculated_sl_price
        self.liquidation_price = self._calculate_liquidation_price(entry_price, self.position_direction)
        self.trade_open_bar_index = entry_bar_index # Store iloc index of entry bar
        # trade_max_holding_bars is already set above

        # --- Update Balance (Deduct Entry Fee) ---
        self.current_balance -= entry_fee
        # Equity is updated at the end of the bar loop in _update_equity_curve

        # --- Log Entry ---
        trade_type = 'LONG' if self.position_direction == 1 else 'SHORT'
        # Correctly format TP and Liquidation prices conditionally
        tp_display = f"{self.current_trade_tp_price:.{self.price_precision}f}" if pd.notna(self.current_trade_tp_price) else 'N/A'
        liq_display = f"{self.liquidation_price:.{self.price_precision}f}" if pd.notna(self.liquidation_price) else 'N/A'
        holding_display = f"{self.trade_max_holding_bars} bars" if self.trade_max_holding_bars is not None else 'None'

        self.logger.info(f"ENTRY @ {entry_timestamp}: Entered {trade_type} | Qty: {self.position_asset_qty:.{self.quantity_precision}f} | Entry: {self.entry_price:.{self.price_precision}f} | Value: {self.position_value_entry_usd:.2f}")
        self.logger.info(f"  SL: {self.current_trade_sl_price:.{self.price_precision}f} | TP: {tp_display} | Est. Liq: {liq_display} | Max Holding: {holding_display}")
        self.logger.info(f"  Entry Fee: {entry_fee:.4f} | New Balance: {self.current_balance:.2f}")


    def _calculate_dynamic_barriers(self, entry_price: float, current_volatility: Optional[float]) -> Tuple[float, float]:
        """Calculates dynamic TP and SL prices based on volatility or fixed percentages."""
        if self.position_direction == 0: # Should not happen if called correctly
             self.logger.error("Attempted to calculate barriers when flat.")
             return np.nan, np.nan
        if pd.isna(entry_price) or entry_price <= FLOAT_EPSILON:
            self.logger.error(f"Invalid entry price ({entry_price}) for barrier calculation.")
            return np.nan, np.nan

        # --- Determine TP/SL Percentages (Fractions) ---
        tp_frac = self.fixed_take_profit_fraction
        sl_frac = self.fixed_stop_loss_fraction

        if self.volatility_adjustment_enabled:
            if pd.isna(current_volatility) or current_volatility <= FLOAT_EPSILON:
                self.logger.warning(f"Volatility adjustment enabled but ATR is invalid ({current_volatility}). Using fixed TP/SL percentages.")
            else:
                try:
                    vol_pct_of_price = current_volatility / entry_price
                    dynamic_tp_frac = self.alpha_take_profit * vol_pct_of_price
                    dynamic_sl_frac = self.alpha_stop_loss * vol_pct_of_price
                    # Use the larger of fixed or dynamic percentage for TP/SL levels
                    tp_frac = max(self.fixed_take_profit_fraction, dynamic_tp_frac)
                    sl_frac = max(self.fixed_stop_loss_fraction, dynamic_sl_frac)
                    self.logger.debug(f"Dynamic barriers calculated: TP%={tp_frac*100:.2f}, SL%={sl_frac*100:.2f} (ATR={current_volatility:.{self.price_precision+1}f})")
                except Exception as e:
                    self.logger.error(f"Error calculating dynamic barrier percentages: {e}. Using fixed.", exc_info=True)
                    # Fallback to fixed on error
                    tp_frac = self.fixed_take_profit_fraction
                    sl_frac = self.fixed_stop_loss_fraction
        else:
             self.logger.debug(f"Using fixed barrier percentages: TP%={tp_frac*100:.2f}, SL%={sl_frac*100:.2f}")


        # Final check on calculated fractions
        if tp_frac <= FLOAT_EPSILON:
             self.logger.warning(f"Calculated TP% ({tp_frac*100:.4f}) is zero or negative. Setting TP to NaN.")
             take_profit_price = np.nan
        else:
             # Calculate TP price based on direction
             take_profit_price = entry_price * (1 + self.position_direction * tp_frac)
             # Apply price precision rounding
             take_profit_price = self._round_price(take_profit_price)
             # Final validation after rounding
             if pd.isna(take_profit_price) or take_profit_price <= FLOAT_EPSILON:
                  self.logger.warning(f"TP price invalid ({take_profit_price}) after rounding. Setting TP to NaN.")
                  take_profit_price = np.nan


        if sl_frac <= FLOAT_EPSILON:
             self.logger.error(f"Calculated SL% ({sl_frac*100:.4f}) is zero or negative. Cannot set SL.")
             stop_loss_price = np.nan # Indicate failure
        else:
             # Calculate SL price based on direction
             stop_loss_price = entry_price * (1 - self.position_direction * sl_frac)
             # Apply price precision rounding
             stop_loss_price = self._round_price(stop_loss_price)
             # Final validation after rounding
             if pd.isna(stop_loss_price) or stop_loss_price <= FLOAT_EPSILON:
                  self.logger.error(f"SL price invalid ({stop_loss_price}) after rounding. Cannot set SL.")
                  stop_loss_price = np.nan # Indicate failure


        self.logger.debug(f"Calculated barrier prices: TP={take_profit_price}, SL={stop_loss_price}")
        return take_profit_price, stop_loss_price


    def _calculate_position_size(self, entry_price: float, stop_loss_price: float) -> Tuple[float, float]:
        """Calculates position size based on risk, equity, SL distance, and margin constraints."""
        if pd.isna(entry_price) or entry_price <= 0 or \
           pd.isna(stop_loss_price) or stop_loss_price <= 0 or \
           self.current_equity <= 0 or self.leverage <= 0 or self.risk_per_trade_fraction <= 0:
            self.logger.warning("Invalid input for position size calculation. Returning zero size.")
            return 0.0, 0.0

        # --- 1. Calculate Risk-Based Size ---
        capital_at_risk = self.current_equity * self.risk_per_trade_fraction
        stop_loss_distance = abs(entry_price - stop_loss_price)

        if stop_loss_distance <= FLOAT_EPSILON:
            self.logger.warning(f"Stop loss distance is too small ({stop_loss_distance:.{self.price_precision+2}f}). Cannot calculate risk-based size.")
            risk_based_qty = 0.0
        else:
            # Risk-based quantity = Capital at Risk / (Stop Loss Distance * Price per Asset)
            # For futures, SL distance is in price units, so quantity is Capital at Risk / SL Distance
            risk_based_qty = capital_at_risk / stop_loss_distance
            self.logger.debug(f"Risk Calc: Equity={self.current_equity:.2f}, RiskAmt={capital_at_risk:.2f}, SLDist={stop_loss_distance:.{self.price_precision}f}, RiskBasedQty={risk_based_qty:.{self.quantity_precision+4}f}")

        # --- 2. Calculate Max Size Allowed by Margin ---
        # Max Position Value = Balance / (Initial Margin Rate + Entry Fee Rate)
        # Note: Using Balance here, as margin is taken from balance, not equity.
        initial_margin_rate = 1.0 / self.leverage
        effective_cost_rate = initial_margin_rate + self.trading_fee_rate

        if effective_cost_rate <= FLOAT_EPSILON:
             self.logger.warning(f"Effective cost rate ({effective_cost_rate:.6f}) non-positive. Cannot calculate max margin size.")
             max_allowed_qty = float('inf') # Effectively no margin limit if calculation fails
        else:
             max_position_value_usd = self.current_balance / effective_cost_rate
             max_allowed_qty = max_position_value_usd / entry_price if entry_price > FLOAT_EPSILON else 0.0
             self.logger.debug(f"Margin Calc: Balance={self.current_balance:.2f}, MaxValue={max_position_value_usd:.2f}, MaxAllowedQty={max_allowed_qty:.{self.quantity_precision+4}f}")

        # --- 3. Determine Final Quantity ---
        # Use the minimum of risk-based and margin-based quantities
        final_asset_qty = min(risk_based_qty, max_allowed_qty)

        # Apply exchange quantity precision rules (rounding down)
        final_asset_qty = self._round_quantity(final_asset_qty)

        if final_asset_qty is None or final_asset_qty <= FLOAT_EPSILON:
             self.logger.warning(f"Quantity rounded to zero or negative ({final_asset_qty}). Setting size to 0.")
             return 0.0, 0.0

        # --- 4. Check Exchange Minimums ---
        final_position_value_usd = final_asset_qty * entry_price

        if final_asset_qty < self.min_quantity - FLOAT_EPSILON: # Use tolerance for comparison
             self.logger.warning(f"Calculated quantity {final_asset_qty:.{self.quantity_precision}f} is below exchange minimum quantity {self.min_quantity:.{self.quantity_precision}f}. Setting size to 0.")
             return 0.0, 0.0
        if final_position_value_usd < self.min_notional - FLOAT_EPSILON: # Use tolerance for comparison
             self.logger.warning(f"Calculated notional value {final_position_value_usd:.2f} is below exchange minimum notional {self.min_notional:.2f}. Setting size to 0.")
             return 0.0, 0.0

        self.logger.debug(f"Final Position Size: Qty={final_asset_qty:.{self.quantity_precision}f}, Value={final_position_value_usd:.2f}")
        return final_asset_qty, final_position_value_usd


    def _calculate_liquidation_price(self, entry_price: float, trade_direction: int) -> float:
        """Estimates liquidation price based on leverage and maintenance margin rate."""
        if pd.isna(entry_price) or entry_price <= 0 or trade_direction == 0 or self.leverage <= 0:
            return np.nan

        try:
            initial_margin_rate = 1.0 / self.leverage
            # Use maintenance margin rate from config
            maint_margin_rate = self.maintenance_margin_rate

            # Check for insufficient initial margin relative to maintenance margin
            if initial_margin_rate <= maint_margin_rate + FLOAT_EPSILON:
                self.logger.warning(f"Initial margin rate ({initial_margin_rate:.4f}) <= maintenance margin rate ({maint_margin_rate:.4f}). Liquidation likely immediate.")
                # Return a price very close to entry in the losing direction
                # This simulates near-immediate liquidation without hitting exactly entry price
                return entry_price * (1 - trade_direction * FLOAT_EPSILON * 100) # Use a slightly larger epsilon for liq sim

            # Simplified liquidation price formula (may vary slightly by exchange)
            # This formula is often used for cross margin or simplified isolated margin
            # Liq Price = Entry * (1 - Direction * (Initial Margin Rate - Maintenance Margin Rate))
            liq_price = entry_price * (1 - trade_direction * (initial_margin_rate - maint_margin_rate))

            # Apply price precision rounding
            liq_price = self._round_price(liq_price)

            # Ensure non-negative and return NaN if invalid after rounding
            return liq_price if pd.notna(liq_price) and liq_price > FLOAT_EPSILON else np.nan

        except Exception as e:
            self.logger.error(f"Error calculating liquidation price: {e}", exc_info=True)
            return np.nan


    def _check_exit_conditions(self, bar_index: int) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Checks exit conditions (Liquidation, SL, TP, Time Limit, Invalid OHLC) for the current bar.
        Returns (exit_flag, exit_reason, exit_price).
        Prioritizes Liquidation -> SL -> TP -> Time Limit -> Invalid OHLC.
        Signal-based exits are handled in the main loop *after* checking these.
        """
        if self.position_direction == 0: return False, None, None # No position to exit

        current_bar = self.data.iloc[bar_index]
        current_open, current_high, current_low, current_close = current_bar['open'], current_bar['high'], current_bar['low'], current_bar['close']

        # Check for invalid price data within the bar
        if pd.isna(current_open) or pd.isna(current_high) or pd.isna(current_low) or pd.isna(current_close):
             self.logger.warning(f"Index {bar_index} ({self.data.index[bar_index]}): Invalid OHLC data. Cannot check exit conditions accurately.")
             # Decide how to handle: potentially force close, or skip checks for this bar
             # Skipping checks might lead to unrealistic scenarios. Forcing close is safer.
             self.logger.warning(f"Forcing position closure due to invalid OHLC data at index {bar_index}.")
             # Use the previous bar's close as a rough estimate if possible, or just NaN
             fallback_price = self.data.iloc[bar_index-1]['close'] if bar_index > 0 else np.nan
             return True, 'invalid_ohlc', fallback_price # Force exit with reason

        # --- 1. Check Liquidation ---
        # Assumes liquidation happens if low/high touches the estimated price.
        # Use tolerance (FLOAT_EPSILON) for floating point comparisons
        if pd.notna(self.liquidation_price):
            if (self.position_direction == 1 and current_low <= self.liquidation_price + FLOAT_EPSILON) or \
               (self.position_direction == -1 and current_high >= self.liquidation_price - FLOAT_EPSILON):
                exit_trade = True
                exit_reason = 'liquidation'
                # Assume liquidation occurs exactly at the liquidation price
                exit_price = self.liquidation_price
                self.logger.debug(f"Index {bar_index}: LIQUIDATION triggered at {exit_price:.{self.price_precision}f}")
                return exit_trade, exit_reason, exit_price # Exit immediately

        # --- 2. Check Stop Loss ---
        # SL triggers if low/high touches or crosses the SL price.
        # Use tolerance (FLOAT_EPSILON) for floating point comparisons
        if pd.notna(self.current_trade_sl_price):
            if (self.position_direction == 1 and current_low <= self.current_trade_sl_price + FLOAT_EPSILON) or \
               (self.position_direction == -1 and current_high >= self.current_trade_sl_price - FLOAT_EPSILON):
                exit_trade = True
                exit_reason = 'stop_loss'
                # Assume SL filled exactly at the SL price
                exit_price = self.current_trade_sl_price
                self.logger.debug(f"Index {bar_index}: STOP LOSS hit at {exit_price:.{self.price_precision}f}")
                # Check if TP was also hit in the same bar (SL takes priority)
                if pd.notna(self.current_trade_tp_price):
                     if (self.position_direction == 1 and current_high >= self.current_trade_tp_price - FLOAT_EPSILON) or \
                        (self.position_direction == -1 and current_low <= self.current_trade_tp_price + FLOAT_EPSILON):
                          self.logger.warning(f"Index {bar_index}: Both SL and TP hit in the same bar. Exiting due to SL (priority).")
                return exit_trade, exit_reason, exit_price # Exit on SL

        # --- 3. Check Take Profit ---
        # TP triggers if low/high touches or crosses the TP price.
        # Use tolerance (FLOAT_EPSILON) for floating point comparisons
        if pd.notna(self.current_trade_tp_price):
            if (self.position_direction == 1 and current_high >= self.current_trade_tp_price - FLOAT_EPSILON) or \
               (self.position_direction == -1 and current_low <= self.current_trade_tp_price + FLOAT_EPSILON):
                exit_trade = True
                exit_reason = 'take_profit'
                # Assume TP filled exactly at the TP price
                exit_price = self.current_trade_tp_price
                self.logger.debug(f"Index {bar_index}: TAKE PROFIT hit at {exit_price:.{self.price_precision}f}")
                return exit_trade, exit_reason, exit_price # Exit on TP

        # --- 4. Check Time Limit (Using trade-specific max holding) ---
        # Use self.trade_max_holding_bars which was set at entry
        if self.trade_max_holding_bars is not None and self.trade_open_bar_index != -1:
            # Holding duration is the number of bars *since* entry, including the current bar.
            # If entered on bar 10, and current bar is 15, duration is 15 - 10 + 1 = 6 bars.
            holding_duration_bars = bar_index - self.trade_open_bar_index + 1
            if holding_duration_bars > self.trade_max_holding_bars: # Exit *after* max_holding_period bars have passed
                exit_trade = True
                exit_reason = 'time_limit'
                # Exit at the close price of the current bar when time limit reached
                exit_price = current_close
                self.logger.debug(f"Index {bar_index}: Max holding period ({self.trade_max_holding_bars} bars) reached for this trade. Exiting at close {exit_price:.{self.price_precision}f}.")
                return exit_trade, exit_reason, exit_price # Exit on Time Limit

        # Invalid OHLC check is now done at the beginning and forces an exit if needed.

        return False, None, None # No hard exit condition met


    def _close_position(self, bar_index: int, exit_reason: str, exit_price: float):
        """Closes the current position, calculates PnL, updates balance/equity, logs trade."""
        if self.position_direction == 0:
            self.logger.error("Attempted to close position when already flat.")
            return
        # Ensure exit price is valid before proceeding
        if pd.isna(exit_price) or exit_price <= FLOAT_EPSILON:
             self.logger.error(f"Invalid exit price ({exit_price}) provided for closing position at bar {bar_index}. Cannot close accurately.")
             # Attempt to use bar close price as a fallback, but log critical warning
             fallback_price = self.data.iloc[bar_index]['close']
             self.logger.critical(f"CRITICAL: Using bar close price {fallback_price:.{self.price_precision}f} as fallback exit price due to invalid input.")
             exit_price = fallback_price # Use fallback
             if pd.isna(exit_price) or exit_price <= FLOAT_EPSILON:
                  self.logger.critical("Fallback exit price (bar close) is also invalid. Cannot close position or calculate PnL.")
                  # Reset state without logging trade or changing balance/equity
                  self._reset_position_state()
                  return

        exit_timestamp = self.data.index[bar_index]
        # Ensure entry_time is not NaT before calculating duration
        entry_timestamp = self.entry_time if pd.notna(self.entry_time) else self.data.index[self.trade_open_bar_index] # Fallback to bar index if entry_time is NaT

        # --- Calculate PnL ---
        price_diff = exit_price - self.entry_price
        gross_pnl = price_diff * self.position_asset_qty * self.position_direction

        # --- Calculate Exit Fee ---
        # Exit fee is based on the notional value at exit
        exit_value_usd = abs(self.position_asset_qty * exit_price)
        # Use potentially higher liquidation fee rate if applicable
        fee_rate_for_close = self.liquidation_fee_rate if exit_reason == 'liquidation' else self.trading_fee_rate
        exit_fee = exit_value_usd * fee_rate_for_close

        # Entry fee was already deducted from balance at entry, but log it here for the trade record
        entry_fee_paid = self.position_value_entry_usd * self.trading_fee_rate

        total_fees = entry_fee_paid + exit_fee

        # --- Calculate Net PnL (for logging and consistency check) ---
        net_pnl = gross_pnl - total_fees

        # --- Update Balance ---
        # Balance increases by Gross PnL minus the Exit Fee (entry fee already deducted)
        balance_change = gross_pnl - exit_fee
        self.current_balance += balance_change

        # --- Update Equity ---
        # After closing, equity equals the new balance
        self.current_equity = self.current_balance

        # --- Log Trade Details ---
        # Ensure trade_open_bar_index is valid before calculating duration
        holding_duration = bar_index - self.trade_open_bar_index + 1 if self.trade_open_bar_index != -1 else 0 # Bars inclusive

        trade_log = {
            'entry_time': entry_timestamp, # Use the stored entry_time
            'exit_time': exit_timestamp,
            'direction': self.position_direction, # Direction of the closed trade
            'entry_price': self._round_price(self.entry_price),
            'exit_price': self._round_price(exit_price),
            'size_qty': self._round_quantity(self.position_asset_qty),
            'size_usd_entry': round(self.position_value_entry_usd, 2) if pd.notna(self.position_value_entry_usd) else np.nan,
            'sl_price': self._round_price(self.current_trade_sl_price),
            'tp_price': self._round_price(self.current_trade_tp_price),
            'liq_price': self._round_price(self.liquidation_price),
            'gross_pnl': round(gross_pnl, 4) if pd.notna(gross_pnl) else np.nan,
            'entry_fee': round(entry_fee_paid, 4) if pd.notna(entry_fee_paid) else np.nan,
            'exit_fee': round(exit_fee, 4) if pd.notna(exit_fee) else np.nan,
            'total_fees': round(total_fees, 4) if pd.notna(total_fees) else np.nan,
            'net_pnl': round(net_pnl, 4) if pd.notna(net_pnl) else np.nan, # Log the calculated net PnL
            'exit_reason': exit_reason,
            'holding_duration_bars': holding_duration,
            'balance_after_trade': round(self.current_balance, 2) if pd.notna(self.current_balance) else np.nan, # Record balance *after* this trade
            'equity_after_trade': round(self.current_equity, 2) if pd.notna(self.current_equity) else np.nan,   # Record equity *after* this trade
            'max_holding_at_entry': self.trade_max_holding_bars # NEW: Log the max holding period used for this trade
        }
        self.trades.append(trade_log)

        # Log closure summary
        trade_type = 'LONG' if self.position_direction == 1 else 'SHORT'
        self.logger.info(f"EXIT @ {exit_timestamp}: Closed {trade_type} | Reason: {exit_reason} | Exit: {exit_price:.{self.price_precision}f} | Net PnL: {net_pnl:.2f}")
        self.logger.info(f"  New Balance: {self.current_balance:.2f} | New Equity: {self.current_equity:.2f}")

        # --- Reset Position State ---
        self._reset_position_state()


    def _update_equity_curve(self, bar_index: int):
        """Updates the equity curve for the current bar index."""
        # Ensure bar_index is valid for the data and equity_curve Series
        if bar_index >= len(self.data) or bar_index >= len(self.equity_curve):
             self.logger.error(f"Attempted to update equity curve at invalid index {bar_index}. Data length: {len(self.data)}, Curve length: {len(self.equity_curve)}")
             return

        current_close = self.data.iloc[bar_index]['close']

        if self.position_direction != 0:
            # Calculate unrealized PnL based on current close
            if pd.notna(current_close) and pd.notna(self.entry_price):
                unrealized_pnl = (current_close - self.entry_price) * self.position_asset_qty * self.position_direction
                self.current_equity = self.current_balance + unrealized_pnl
            else:
                # If close or entry price invalid, carry forward previous equity
                if bar_index > 0:
                     # Ensure previous equity exists before accessing
                     if pd.notna(self.equity_curve.iloc[bar_index - 1]):
                          self.current_equity = self.equity_curve.iloc[bar_index - 1]
                          self.logger.warning(f"Invalid price data at index {bar_index}. Carrying forward previous equity ({self.current_equity:.2f}).")
                     else:
                           # Fallback if previous equity is also somehow invalid
                           self.current_equity = self.current_balance
                           self.logger.warning(f"Invalid price data and previous equity at index {bar_index}. Setting equity to balance ({self.current_equity:.2f}).")

                else: # First bar edge case
                     self.current_equity = self.current_balance
                     self.logger.warning(f"Invalid price data on first bar index {bar_index}. Setting equity to balance ({self.current_equity:.2f}).")

        else: # Flat
            self.current_equity = self.current_balance

        # Record equity for the current bar
        self.equity_curve.iloc[bar_index] = self.current_equity


    def _calculate_summary_metrics(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates summary performance metrics from completed trades."""
        self.logger.info("Calculating summary metrics...")
        metrics = {"initial_capital": self.initial_capital}

        # --- Equity Curve Metrics ---
        if not self.equity_curve.empty:
            metrics["final_equity"] = self.equity_curve.iloc[-1]
            metrics["total_return_pct"] = ((metrics["final_equity"] / self.initial_capital) - 1) * 100 if self.initial_capital > FLOAT_EPSILON else 0.0
            metrics["peak_equity"] = self.equity_curve.max()
            # Calculate drawdown relative to cumulative maximum equity
            cumulative_max = self.equity_curve.cummax()
            # Avoid division by zero if cumulative_max is zero (shouldn't happen with positive initial capital)
            drawdown = (self.equity_curve - cumulative_max) / cumulative_max.replace(0, np.nan)
            metrics["max_drawdown_pct"] = abs(drawdown.min()) * 100 if pd.notna(drawdown.min()) else 0.0
            metrics["equity_change"] = metrics["final_equity"] - metrics["initial_capital"] # For consistency check
        else:
            self.logger.warning("Equity curve is empty. Cannot calculate equity-based metrics.")
            metrics.update({
                "final_equity": self.initial_capital, "total_return_pct": 0.0,
                "peak_equity": self.initial_capital, "max_drawdown_pct": 0.0,
                "equity_change": 0.0
            })

        # --- Trade Metrics ---
        num_trades = len(self.trades)
        metrics["num_trades"] = num_trades

        if num_trades > 0:
            trades_df = pd.DataFrame(self.trades)
            # Ensure 'net_pnl' column exists before summing
            metrics["total_net_pnl"] = trades_df['net_pnl'].sum() if 'net_pnl' in trades_df.columns else 0.0
            metrics["total_gross_pnl"] = trades_df['gross_pnl'].sum() if 'gross_pnl' in trades_df.columns else 0.0
            metrics["total_fees"] = trades_df['total_fees'].sum() if 'total_fees' in trades_df.columns else 0.0

            # Count wins/losses based on net_pnl > 0 (excluding zero PnL trades as losses/break-even)
            if 'net_pnl' in trades_df.columns:
                 metrics["num_wins"] = (trades_df['net_pnl'] > FLOAT_EPSILON).sum()
                 metrics["num_losses"] = (trades_df['net_pnl'] <= FLOAT_EPSILON).sum() # Includes break-even
            else:
                 metrics["num_wins"] = 0
                 metrics["num_losses"] = num_trades # Assume all are losses if PnL is missing

            metrics["win_rate_pct"] = (metrics["num_wins"] / num_trades) * 100 if num_trades > 0 else 0.0
            metrics["avg_pnl_per_trade"] = metrics["total_net_pnl"] / num_trades if num_trades > 0 else np.nan
            metrics["avg_win_pnl"] = trades_df.loc[trades_df['net_pnl'] > FLOAT_EPSILON, 'net_pnl'].mean() if 'net_pnl' in trades_df.columns else np.nan
            metrics["avg_loss_pnl"] = trades_df.loc[trades_df['net_pnl'] <= FLOAT_EPSILON, 'net_pnl'].mean() if 'net_pnl' in trades_df.columns else np.nan

            # Profit Factor: Gross Wins / Absolute Gross Losses
            # Sum gross profits from winning trades
            total_wins_gross = trades_df.loc[trades_df['gross_pnl'] > FLOAT_EPSILON, 'gross_pnl'].sum() if 'gross_pnl' in trades_df.columns else 0.0
            # Sum absolute gross losses from losing trades (including break-even)
            total_losses_gross = abs(trades_df.loc[trades_df['gross_pnl'] <= FLOAT_EPSILON, 'gross_pnl'].sum()) if 'gross_pnl' in trades_df.columns else 0.0
            metrics["profit_factor"] = total_wins_gross / total_losses_gross if total_losses_gross > FLOAT_EPSILON else np.inf # Handle division by zero

            metrics["avg_holding_duration_bars"] = trades_df['holding_duration_bars'].mean() if 'holding_duration_bars' in trades_df.columns else np.nan
            # Add Sharpe Ratio, Sortino Ratio, etc. if desired (requires risk-free rate and potentially daily returns)
        else:
            metrics.update({
                "total_net_pnl": 0.0, "total_gross_pnl": 0.0, "total_fees": 0.0,
                "num_wins": 0, "num_losses": 0, "win_rate_pct": 0.0,
                "avg_pnl_per_trade": np.nan, "avg_win_pnl": np.nan, "avg_loss_pnl": np.nan,
                "profit_factor": np.nan, "avg_holding_duration_bars": np.nan
            })

        # --- Add Configuration Details to Metrics ---
        metrics['config_symbol'] = self.symbol
        metrics['config_interval'] = self.interval
        metrics['config_model_type'] = self.model_type
        metrics['config_leverage'] = self.leverage
        metrics['config_risk_per_trade_pct'] = self.risk_per_trade_pct # Log original percentage
        metrics['config_trading_fee_rate'] = self.trading_fee_rate
        metrics['config_maintenance_margin_rate'] = self.maintenance_margin_rate # Log maintenance margin rate
        metrics['config_liquidation_fee_rate'] = self.liquidation_fee_rate # Log liquidation fee rate

        metrics['config_volatility_adjustment_enabled'] = self.volatility_adjustment_enabled
        if self.volatility_adjustment_enabled:
             metrics['config_volatility_window_bars'] = self.volatility_window_bars
             metrics['config_fixed_take_profit_pct'] = self.fixed_take_profit_pct
             metrics['config_fixed_stop_loss_pct'] = self.fixed_stop_loss_pct
             metrics['config_alpha_take_profit'] = self.alpha_take_profit
             metrics['config_alpha_stop_loss'] = self.alpha_stop_loss
        else: # Log fixed TP/SL even if volatility adjustment is disabled
             metrics['config_fixed_take_profit_pct'] = self.fixed_take_profit_pct
             metrics['config_fixed_stop_loss_pct'] = self.fixed_stop_loss_pct
             # Log N/A for alpha values if disabled
             metrics['config_alpha_take_profit'] = np.nan
             metrics['config_alpha_stop_loss'] = np.nan


        metrics['config_trend_filter_enabled'] = self.trend_filter_enabled
        if self.trend_filter_enabled:
             metrics['config_trend_filter_ema_period'] = self.trend_filter_ema_period
        else: # Log N/A for EMA period if disabled
             metrics['config_trend_filter_ema_period'] = np.nan

        # Log default max holding, and regime-specific if enabled
        metrics['config_default_max_holding_period_bars'] = self.max_holding_period
        metrics['config_volatility_regime_filter_enabled'] = self.volatility_regime_filter_enabled
        if self.volatility_regime_filter_enabled:
             metrics['config_volatility_regime_max_holding_bars'] = self.volatility_regime_max_holding_bars
             metrics['config_allow_trading_in_volatility_regime'] = self.allow_trading_in_volatility_regime


        metrics['config_min_quantity'] = self.min_quantity # Log exchange minimums
        metrics['config_min_notional'] = self.min_notional
        metrics['config_tie_breaker'] = self.tie_breaker # Log tie breaker
        metrics['config_exit_on_neutral_signal'] = self.exit_on_neutral_signal
        # Include the new parameters
        metrics['config_allow_long_trades'] = self.allow_long_trades
        metrics['config_allow_short_trades'] = self.allow_short_trades
        # Include confidence filter parameters in metrics
        metrics['config_confidence_filter_enabled'] = self.confidence_filter_enabled
        metrics['config_confidence_threshold_long_pct'] = self.confidence_threshold_long * 100.0 # Log as percentage
        metrics['config_confidence_threshold_short_pct'] = self.confidence_threshold_short * 100.0 # Log as percentage


        self.logger.info("Performance metrics calculated.")
        # Log key metrics
        self.logger.info(f"  Total Return: {metrics.get('total_return_pct', 0.0):.2f}%")
        self.logger.info(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0.0):.2f}%")
        self.logger.info(f"  Win Rate: {metrics.get('win_rate_pct', 0.0):.2f}% ({metrics.get('num_wins', 0)} Wins / {metrics.get('num_losses', 0)} Losses)")
        self.logger.info(f"  Profit Factor: {metrics.get('profit_factor', np.nan):.2f}")
        self.logger.info(f"  Total Net PnL: {metrics.get('total_net_pnl', 0.0):.2f}")

        return metrics


    def _check_pnl_consistency(self, metrics: Dict[str, Any]):
        """Checks if the final equity change matches the sum of net PnLs from trades."""
        equity_change = metrics.get('equity_change', np.nan)
        total_net_pnl_from_trades = metrics.get('total_net_pnl', np.nan)

        # Only perform check if both values are available (not NaN)
        if pd.notna(equity_change) and pd.notna(total_net_pnl_from_trades):
            discrepancy = abs(equity_change - total_net_pnl_from_trades)
            # Define a tolerance for floating point comparison
            # A small absolute tolerance or a relative tolerance based on initial capital
            tolerance = max(self.initial_capital * 1e-6, 1e-4) # Example: 0.0001% of capital or 0.0001 units

            if discrepancy > tolerance:
                self.logger.critical(f"CRITICAL PNL DISCREPANCY DETECTED!")
                self.logger.critical(f"  Equity Change (Final - Initial): {equity_change:.6f}")
                self.logger.critical(f"  Sum of Trade Net PnLs:         {total_net_pnl_from_trades:.6f}")
                self.logger.critical(f"  Discrepancy:                    {discrepancy:.6f}")
                metrics["PnL Consistency Check"] = f"FAIL (Discrepancy: {discrepancy:.6f})"
            else:
                self.logger.info(f"PnL Consistency Check Passed (Discrepancy: {discrepancy:.6f})")
                metrics["PnL Consistency Check"] = "PASS"
        else:
            self.logger.warning("Could not perform PnL consistency check due to missing metrics.")
            metrics["PnL Consistency Check"] = "Unavailable"


    def _save_results(self, trades_df: pd.DataFrame, equity_curve: pd.Series, metrics: Dict[str, Any]):
        """Saves trades, equity curve, and metrics to files based on PATHS config."""
        self.logger.info("Saving backtest results...")
        results_dir = self.paths.get("backtesting_results_dir")
        if not results_dir or not isinstance(results_dir, (str, Path)):
            self.logger.error("Cannot save results: 'backtesting_results_dir' invalid or missing in paths config.")
            return
        results_dir = Path(results_dir) # Ensure it's a Path object
        results_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists

        # Get file patterns from paths config, providing defaults
        trades_pattern = str(self.paths.get("backtesting_trades_pattern", "{symbol}_{interval}_{model_type}_trades.parquet"))
        equity_pattern = str(self.paths.get("backtesting_equity_pattern", "{symbol}_{interval}_{model_type}_equity.parquet"))
        metrics_pattern = str(self.paths.get("backtesting_metrics_pattern", "{symbol}_{interval}_{model_type}_metrics.json"))

        # Prepare parameters for formatting filenames
        file_params = {
            "symbol": self.symbol.replace('/', ''), # Remove '/' from symbol for filename safety
            "interval": self.interval.replace(':', '_'), # Replace ':' from interval for filename safety
            "model_type": self.model_type
        }

        try:
            # --- Save Trades ---
            if self.save_trades and not trades_df.empty:
                trades_path = results_dir / trades_pattern.format(**file_params)
                # Ensure datetime columns are timezone-aware (UTC) before saving
                for col in ['entry_time', 'exit_time']:
                     if col in trades_df.columns and pd.api.types.is_datetime64_any_dtype(trades_df[col]):
                          # Check if timezone-naive, if so localize to UTC
                          if trades_df[col].dt.tz is None:
                               trades_df[col] = trades_df[col].dt.tz_localize('UTC')
                          # If timezone-aware but not UTC, convert to UTC
                          elif str(trades_df[col].dt.tz) != 'UTC':
                               trades_df[col] = trades_df[col].dt.tz_convert('UTC')

                trades_df.to_parquet(trades_path, index=False)
                self.logger.info(f"Trades data saved to {trades_path}")
            elif self.save_trades: self.logger.info("No trades executed to save.")

            # --- Save Equity Curve ---
            if self.save_equity_curve and not equity_curve.empty:
                equity_path = results_dir / equity_pattern.format(**file_params)
                # Ensure index is timezone-aware (UTC)
                if equity_curve.index.tz is None:
                     equity_curve.index = equity_curve.index.tz_localize('UTC')
                elif str(equity_curve.index.tz) != 'UTC':
                     equity_curve.index = equity_curve.index.tz_convert('UTC')

                equity_curve.to_frame(name='equity').to_parquet(equity_path)
                self.logger.info(f"Equity curve data saved to {equity_path}")
            elif self.save_equity_curve: self.logger.info("Equity curve is empty, not saving.")

            # --- Save Metrics ---
            if self.save_metrics and metrics:
                metrics_path = results_dir / metrics_pattern.format(**file_params)
                # Custom serializer for JSON compatibility (handles numpy types, Timestamps, Path, NaN/Inf)
                def default_serializer(obj):
                    if isinstance(obj, (np.integer, np.int64)): return int(obj)
                    if isinstance(obj, (np.floating, np.float_)):
                         if np.isnan(obj): return "NaN"
                         if np.isinf(obj): return "Infinity" if obj > 0 else "-Infinity"
                         return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if isinstance(obj, pd.Timestamp): return obj.isoformat()
                    if isinstance(obj, Path): return str(obj)
                    try:
                        # Attempt standard JSON serialization first
                        return json.JSONEncoder().default(obj)
                    except TypeError:
                        self.logger.warning(f"Cannot serialize type {type(obj)} for value {obj}. Converting to string.")
                        return str(obj)

                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4, default=default_serializer)
                self.logger.info(f"Summary metrics saved to {metrics_path}")
            elif self.save_metrics: self.logger.info("No metrics calculated to save.")

        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}", exc_info=True)


    def _reset_position_state(self):
        """Resets internal state variables related to an open position."""
        self.position_direction = 0
        self.entry_price = np.nan
        self.entry_time = pd.NaT # Use pandas NaT for datetime
        self.position_value_entry_usd = 0.0
        self.position_asset_qty = 0.0
        self.liquidation_price = np.nan
        self.current_trade_sl_price = np.nan
        self.current_trade_tp_price = np.nan
        self.trade_open_bar_index = -1
        self.trade_max_holding_bars = None # NEW: Reset trade-specific max holding


    # --- Helper methods for rounding ---
    def _round_price(self, price: Optional[float]) -> Optional[float]:
        """Rounds a price to the configured precision."""
        if pd.isna(price): return np.nan
        # Ensure price_precision is a non-negative integer
        if not isinstance(self.price_precision, int) or self.price_precision < 0:
             self.logger.warning(f"Invalid price_precision: {self.price_precision}. Cannot round price.")
             return price # Return original price if precision is invalid

        try:
            return round(price, self.price_precision)
        except (TypeError, ValueError) as e:
             self.logger.warning(f"Could not round price {price} to precision {self.price_precision}: {e}")
             return np.nan # Return NaN if rounding fails

    def _round_quantity(self, quantity: Optional[float]) -> Optional[float]:
        """Rounds a quantity DOWN to the configured precision (step size)."""
        if pd.isna(quantity): return np.nan
        # Ensure quantity_precision is a non-negative integer
        if not isinstance(self.quantity_precision, int) or self.quantity_precision < 0:
             self.logger.warning(f"Invalid quantity_precision: {self.quantity_precision}. Cannot round quantity.")
             return quantity # Return original quantity if precision is invalid

        try:
            factor = 10 ** self.quantity_precision
            # Floor division equivalent for floating point precision
            # Multiply by factor, floor, then divide by factor
            return math.floor(quantity * factor) / factor
        except (TypeError, ValueError) as e:
             self.logger.warning(f"Could not round quantity {quantity} to precision {self.quantity_precision}: {e}")
             return np.nan # Return NaN if rounding fails
