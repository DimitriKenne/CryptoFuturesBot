# config/params.py

"""
Central configuration file for the trading bot project.

Stores parameters for general settings, exchange interactions, feature engineering,
labeling, model training, strategy logic, backtesting, and notifications.

Sensitive information (API keys/secrets, tokens) should be loaded from
environment variables for security. Use clear comments to explain each parameter
and its expected data type or format.

**MODIFIED (NEW)**: Added parameters for confidence threshold filtering in STRATEGY_CONFIG.
**MODIFIED (NEW)**: Added parameters for volatility regime filtering in STRATEGY_CONFIG.
**MODIFIED (NEW)**: Added parameters for PCA dimensionality reduction in MODEL_CONFIG.
"""

from typing import Optional
import numpy as np
import os
from pathlib import Path
import logging


# --- Import dotenv to load environment variables FIRST ---
from dotenv import load_dotenv, find_dotenv
# Load environment variables from .env file
load_dotenv(find_dotenv())
# --- End dotenv import ---

# --- Import for Hyperparameter Tuning Distributions ---
# Need to install scipy: pip install scipy
try:
    from scipy.stats import uniform, randint
    SCIPY_AVAILABLE = True
except ImportError:
    uniform = None
    randint = None
    SCIPY_AVAILABLE = False
    logging.warning("Scipy not found. Hyperparameter tuning distributions (uniform, randint) will not be available.")

# --- Conditional Import for TensorFlow/Keras for LSTM Availability Check ---
# Need to install tensorflow: pip install tensorflow
try:
    import tensorflow as tf
    tf_version = getattr(tf, '__version__', 'unknown')
    LSTM_AVAILABLE = True
    # Optionally log TensorFlow version and GPU availability here if desired,
    # but it might be better to do this in scripts that actively use TF (like model_trainer).
    # logging.info(f"TensorFlow (version {tf_version}) imported successfully in params.py.")
    # if tf.config.list_physical_devices('GPU'):
    #     logging.info("GPU is available for TensorFlow.")
    # else:
    #     logging.info("GPU is not available for TensorFlow.")
except ImportError:
    tf = None
    LSTM_AVAILABLE = False
    logging.warning("TensorFlow not found. LSTM model type configuration checks will be based on availability=False.")
except Exception as e:
    # Catch other potential errors during TF import (e.g., DLL issues)
    tf = None
    LSTM_AVAILABLE = False
    logging.error(f"Error importing TensorFlow/Keras in params.py: {e}", exc_info=True)


# --- Logger Setup ---
# Basic logger for messages generated within this config file (e.g., missing env vars)
logger = logging.getLogger(__name__)

# --- Project Root ---
# Assumes this file is located at project_root/config/params.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Environment Variable Loading ---
# Load sensitive keys from environment variables. Provide clear warnings if missing.
def get_env_var(var_name: str, is_required: bool = True, default: str = None) -> Optional[str]:
    """Helper function to get environment variables with logging."""
    value = os.getenv(var_name)
    if not value:
        if is_required and default is None:
            logger.critical(f"CRITICAL: Required environment variable '{var_name}' is not set.")
            # In a real application, you might exit here or raise an error
            # raise ValueError(f"Missing required environment variable: {var_name}")
            return None # Return None to allow checks later
        elif default is not None:
             logger.warning(f"Environment variable '{var_name}' not set. Using default value.")
             return default
        else:
            logger.info(f"Optional environment variable '{var_name}' is not set.")
            return None
    return value

# Load API Keys and Secrets
BINANCE_API_KEY = get_env_var("BINANCE_API_KEY", is_required=True)
BINANCE_API_SECRET = get_env_var("BINANCE_API_SECRET", is_required=True)
# Load Notification Credentials
TELEGRAM_BOT_TOKEN = get_env_var("TELEGRAM_BOT_TOKEN", is_required=False) # Required only if Telegram enabled
TELEGRAM_CHAT_ID = get_env_var("TELEGRAM_CHAT_ID", is_required=False)   # Required only if Telegram enabled
# Add other sensitive variables here (e.g., SLACK_WEBHOOK_URL)
# SLACK_WEBHOOK_URL = get_env_var("SLACK_WEBHOOK_URL", is_required=False)


# --- General Configuration ---
GENERAL_CONFIG = {
    "random_seed": 42,       # int: Seed for random number generators (reproducibility).
    "parallel_jobs": -1,     # int: Number of CPU cores for parallel tasks (-1 = use all).
    "train_split_ratio": 0.8, # float (0.0 to 1.0): Default train/test split ratio.
    "tscv_splits": 5,        # int: Default number of splits for Time Series Cross-Validation in tuning.
    "random_search_n_iter": 20, # int: Default number of iterations for RandomizedSearchCV.
    "tuning_scoring_metric": 'balanced_accuracy', # str: Default metric for tuning ('accuracy', 'balanced_accuracy', 'f1_macro', etc.).
    "cv_jobs":-1,           # int: Number of parallel jobs for cross-validation (-1 = use all).
    
    # --- Monte Carlo Simulation Parameters (NEW) ---
    "monte_carlo_simulations": 0, # int: Number of Monte Carlo simulations to run.
    "monte_carlo_plot_num_curves": 2, # int: Number of equity curves to plot in MC visualization.
}

# --- Exchange Configuration ---
# Parameters for specific exchanges, including API details and adapter settings.
EXCHANGE_CONFIG = {
    "binance_futures": {
        # Credentials (loaded from environment variables above)
        "api_key": BINANCE_API_KEY,
        "api_secret": BINANCE_API_SECRET,

        # Connection Settings
        "testnet": False,        # bool: True for Testnet, False for Production.
        "tld": "com",            # str: Top-Level Domain ('com', 'us', etc.).
        "api_timeout_sec": 30,   # int: API request timeout in seconds.

        # Default Fallbacks for Adapter (if live info fetch fails)
        # Values should be reasonable defaults for common symbols.
        "quantity_precision": 1, # int: Default decimal places for order quantity.
        "price_precision": 4,    # int: Default decimal places for order/ticker price (e.g., ADAUSDT).
        "min_quantity": 0.1,     # float: Default minimum order quantity.
        "min_notional": 5.0,     # float: Default minimum order value (price * quantity).
    },
    # --- Configuration for other exchanges would go here ---
    # "bybit_futures": { ... },
}


# --- Feature Engineering Configuration ---
# Settings for calculating technical indicators and other features.
FEATURE_CONFIG = {
    # Technical Indicator Periods (Use lists for multiple periods where beneficial)
    'sma_periods': [10, 20, 50, 100],   # Simple Moving Averages (Standard range for context)
    'ema_periods': [10, 14, 20, 50, 100, 200], # Exponential Moving Averages (Added 200 for longer-term context)
    'rsi_periods': [7, 14, 28, 50],        # Relative Strength Index (Shorter to medium-long for 5m)
    'bollinger_periods': [20, 30, 40],     # Bollinger Bands (Range around common values)
    'atr_periods': [5, 14, 20, 50],           # Average True Range (Standard and variants for volatility)
    'stochastic_periods': [14, 28],      # Stochastic Oscillator %K (Standard and double)
    'ao_periods': [5, 34],                  # Awesome Oscillator (Standard periods)
    'cci_periods': [14, 20, 40],           # Commodity Channel Index (Faster, standard, slower)
    'mfi_periods': [14, 28],             # Money Flow Index (Standard and double)
    'volume_periods': [10, 20, 30],        # Period for Volume-based indicators (e.g., CMF, OBV calculation window if used)

    # Other Feature Settings
    'support_resistance_periods': [30, 50, 100], # list[int]: Lookback for simple S/R levels (Intraday relevant ranges).
    'candlestick_patterns': [               # list[str]: Patterns to detect (uses talib).
        'hammer', 'engulfing', 'doji', 'evening_star', 'morning_star',
        'harami', 'shooting_star', 'dark_cloud_cover', 'piercing_pattern'
    ],
    'fvg_lookback_bars': 2,                 # int: Lookback for Fair Value Gap detection (standard 3-candle is i vs i-2).
    'z_score_periods': [20, 30, 40],        # list[int]: Periods for Z-score calculation.
    'adr_periods': [14, 28],               # list[int]: Average 5m Range period (different lookbacks for recent volatility).

    # Derived Features
    'trend_strength_periods': [20, 50],     # list[int]: Short/long periods for trend strength (based on SMAs).

    # Temporal Safety Validation (for detecting lookahead bias during feature engineering)
    'temporal_validation': {
        'enabled': True,                    # bool: Run validation check?
        'warning_correlation_threshold': 0.3, # float (0-1): Correlation threshold for warning.
        'error_correlation_threshold': 0.5,   # float (0-1): Correlation threshold for error.
    },

    # Sequence Length (MUST match model and strategy config if using sequence models like LSTM)
    'sequence_length_bars': 5,             # int: Number of past bars for sequence input.
}


# --- Labeling Configuration ---
# Rules for generating target labels for model training.
LABELING_CONFIG = {
    # Strategy Selection
    # str: Supported types: 'directional_ternary', 'triple_barrier', 'max_return_quantile', 'ema_return_percentile'.
    # Choose ONE strategy to be active for the next labeling run
    'label_type': 'ema_return_percentile', # Example: using the new strategy

    # --- Common Parameters ---
    # int: Minimum bars a non-neutral label must persist (used for smoothing). Set to 1 to disable.
    'min_holding_period': 1, # Keep default or adjust slightly (e.g., 3-7) for 5m noise filtering

    # --- Parameters for 'directional_ternary' ---
    # int: How many bars ahead to look for price change.
    'forward_window_bars': 20, # 5 hours lookahead for 5m data
    # float: % price change threshold for Buy (1) / Sell (-1). Adjust based on ADAUSDT 5m swings.
    'price_threshold_pct': 1.5, # Adjusted slightly lower as a starting point

    # --- Parameters for 'triple_barrier' ---
    # int: Max bars to hold before assigning neutral label if no barrier hit (time barrier).
    'max_holding_bars': 100, # Align with forward window
    # float: Fixed take profit percentage (e.g., 1.5 for 1.5%). MUST BE > 0 if use_volatility_adjustment is False.
    # Can be None or non-negative if use_volatility_adjustment is True (acts as minimum).
    'fixed_take_profit_pct': 3, # Starting point for fixed/minimum TP
    # float: Fixed stop loss percentage (e.g., 0.75 for 0.75%). MUST BE > 0 if use_volatility_adjustment is False.
    # Can be None or non-negative if use_volatility_adjustment is True (acts as minimum).
    'fixed_stop_loss_pct': 1, # Starting point for fixed/minimum SL
    # bool: If True, dynamically adjust TP/SL based on ATR (using alphas below). If False, use only fixed percentages.
    'use_volatility_adjustment': True, # Recommended for crypto volatility
    # int: Lookback for ATR calculation (used if use_volatility_adjustment is True).
    #      Feature Engineering MUST generate an ATR column named 'atr_{vol_adj_lookback}'
    #      (e.g., 'atr_14') if this strategy is used with use_volatility_adjustment=True.
    'vol_adj_lookback': 20, # Chosen from FEATURE_CONFIG['atr_periods'] [14, 20, 50] for responsiveness
    # float: ATR multiplier for dynamic TP barrier (used if use_volatility_adjustment is True).
    'alpha_take_profit': 9, # ATR multiplier for TP
    # float: ATR multiplier for dynamic SL barrier (used if use_volatility_adjustment is True).
    'alpha_stop_loss': 3, # ATR multiplier for SL

    # --- Parameters for 'max_return_quantile' ---
    # int: The number of bars to look ahead to find the maximum potential move.
    'quantile_forward_window_bars': 60, # Align with other forward windows
    # float (0-100): The percentile value used to determine the threshold for a "significant" maximum return.
    'quantile_threshold_pct': 60.0, # Standard starting point for selectivity

    # --- Parameters for 'ema_return_percentile' ---
    # Added configuration for the new strategy
    'f_window': 2, # int: Forward window for future close
    'b_window': 25, # int: Backward window for EMA
    'fee': 0.0005, # float: Trading fee (e.0.0005 for 0.05%)
    'beta_increment': 0.1, # float: Increment factor for beta threshold per f_window
    'lower_percentile': 85, # float (0-100): Percentile for alpha threshold
    'upper_percentile': 99.9, # float (0-100): Percentile for beta threshold

}

# --- Model Training Configuration ---
# Hyperparameters and settings for different ML models.
MODEL_CONFIG = {
    "random_forest": {
        "model_type": "random_forest",      # str: Identifier for this model type.
        "params": {                         # dict: Model-specific hyperparameters.
            "n_estimators": 200,            # int: Number of trees.
            "max_depth": None,              # int or None: Max depth of trees.
            "min_samples_split": 2,         # int: Min samples to split a node.
            "class_weight": None, # str or None: Handle class imbalance.
            # Add other scikit-learn RandomForestClassifier params here.
        },
        # Data Handling for Training
        "train_split_ratio": 0.8,           # float (0-1): Train/test split ratio.
        "cv_n_splits": 5,                   # int: Folds for Time Series Cross-Validation.
        "class_balancing": "undersampling",            # str or None: 'undersampling', 'oversampling'. Applied before training.
        "undersample_ratio": 1,           # float (0-1): Target ratio for undersampling.

        # --- Dimensionality Reduction (PCA) ---
        'dimensionality_reduction': {
            'enabled': False,               # bool: Enable PCA for this model?
            'method': 'pca',                # str: Method (currently 'pca' supported).
            'params': {
                'n_components': 0.95,       # float (0-1) for variance explained, or int for number of components.
                                            # If None, all components are kept.
            }
        },
        # --- End of Dimensionality Reduction ---

        # --- Hyperparameter Tuning Distributions for RandomForest ---
        # Requires scipy.stats (uniform, randint)
        'tuning_param_dist': {
            # Prefix parameter names with 'model__' for pipeline tuning
            'model__n_estimators': randint(100, 500),
            'model__max_depth': [3, 5, 7, 10, None], # Include None for no max depth
            'model__min_samples_split': randint(2, 20),
            'model__min_samples_leaf': randint(1, 20),
            'model__class_weight': ['balanced', 'balanced_subsample', None],
            'model__criterion': ['gini', 'entropy'],
            # Add other parameters you want to tune
        } if SCIPY_AVAILABLE else {}, # Only include if Scipy is available
        # --- End of Tuning Distributions ---
    },
    "xgboost": {
        "model_type": "xgboost",            # str: Identifier.
        "params": {                         # dict: XGBoost hyperparameters.
            "n_estimators": 100,            # int: Number of boosting rounds.
            "max_depth": 3,                # int: Max tree depth.
            "learning_rate": 0.1,          # float: Step size shrinkage.
            "subsample": 0.8,              # float (0-1): Fraction of samples per tree.
            "colsample_bytree": 0.8,       # float (0-1): Fraction of features per tree.
            "objective": "multi:softmax",   # str: Objective for multi-class classification.
            "num_class": 3,                 # int: Number of classes (-1, 0, 1).
            "eval_metric": "mlogloss",      # str: Evaluation metric.
            # "use_label_encoder": False,     # bool: Suppress XGBoost warning.
            # Add other XGBoost params (gamma, reg_alpha, reg_lambda, etc.).
        },
        # Data Handling
        "train_split_ratio": 0.8,           # float (0-1)
        "cv_n_splits": 5,                   # int
        "class_balancing": "undersampling",            # str or None
        "undersample_ratio": 1,           # float (0-1)

        # --- Dimensionality Reduction (PCA) ---
        'dimensionality_reduction': {
            'enabled': False,               # bool: Enable PCA for this model?
            'method': 'pca',                # str: Method (currently 'pca' supported).
            'params': {
                'n_components': 0.95,       # float (0-1) for variance explained, or int for number of components.
                                            # If None, all components are kept.
            }
        },
        # --- End of Dimensionality Reduction ---

        # --- Hyperparameter Tuning Distributions for XGBoost ---
        # Requires scipy.stats (uniform, randint)
        'tuning_param_dist': {
            # Prefix parameter names with 'model__' for pipeline tuning
            'model__n_estimators': randint(100, 500),
            'model__learning_rate': uniform(loc=0.01, scale=0.3),
            'model__max_depth': randint(3, 15),
            'model__subsample': uniform(loc=0.6, scale=0.4),
            'model__colsample_bytree': uniform(loc=0.6, scale=0.4),
            'model__gamma': uniform(loc=0, scale=0.5),
            'model__reg_alpha': uniform(loc=0, scale=1),
            'model__reg_lambda': uniform(loc=0.1, scale=1),
            # Add other parameters you want to tune
        } if SCIPY_AVAILABLE else {}, # Only include if Scipy is available
        # --- End of Tuning Distributions ---
    },
    "lstm": {
        "model_type": "lstm",               # str: Identifier.
        "params": {                         # dict: Keras LSTM hyperparameters.
            "sequence_length_bars": 5,     # int: Input sequence length (MUST match FEATURE_CONFIG).
            "n_layers": 3,                  # int: Number of LSTM layers.
            "units_per_layer": 50,          # int: LSTM units per layer.
            "dropout_rate": 0.2,            # float (0-1): Dropout regularization.
            "learning_rate": 0.001,        # float: Optimizer learning rate.
            "epochs": 15,                    # int: Training epochs.
            "batch_size": 32,               # int: Training batch size.
            # Add other Keras params (activation, optimizer, loss, etc.).
        },
        # Data Handling (LSTM often needs separate validation set)
        "train_split_ratio": 0.8,           # float (0-1): Train split.
        "validation_split_ratio": 0.1,      # float (0-1): Validation split (remaining is test).
        "cv_n_splits": 5,                   # int
        "class_balancing": "undersampling",            # str or None: SMOTE, etc. (use with care for sequences).
        "class_weight": None,               # dict or None: Assign weights to classes during training.

        # --- Dimensionality Reduction (PCA) ---
        'dimensionality_reduction': {
            'enabled': False,               # bool: Enable PCA for this model?
            'method': 'pca',                # str: Method (currently 'pca' supported).
            'params': {
                'n_components': 0.95,       # float (0-1) for variance explained, or int for number of components.
                                            # If None, all components are kept.
            }
        },
        # --- End of Dimensionality Reduction ---
    }
    # Add configurations for other model types (e.g., LightGBM, CatBoost, SVM).
}

# --- Live Trading & Backtesting Strategy Configuration ---
# Defines the core trading logic, risk management, and execution rules.
# Used by the live bot and as defaults for the backtester.
STRATEGY_CONFIG = {
    # Core Identification
    "symbol": "ADAUSDT",                # str: Trading symbol (e.g., 'BTCUSDT').
    "interval": "5m",                   # str: Candlestick interval (e.g., '1m', '5m', '1h').
    "model_type": "xgboost",      # str: Model to use ('random_forest', 'xgboost', 'lstm'). MUST match a key in MODEL_CONFIG.

    # Capital and Risk
    "initial_capital": 10.0,          # float: Starting capital for simulation/live tracking.
    "risk_per_trade_pct": 1.0,          # float (0-100): Max percentage of capital to risk per trade.
    "leverage": 5,                     # int: Exchange leverage setting.
    "min_liq_distance_pct": 1.0,        # float (0-100): Minimum required distance (%) between SL and estimated liquidation price.

    # Execution Costs
    "trading_fee_rate": 0.0005,         # float (fraction): Estimated fee per trade side (e.g., 0.0005 = 0.05%).
    "slippage_tolerance_pct": 0.01,     # float (fraction): Assumed slippage for market orders (e.g., 0.01 = 0.01%). Not directly used in order placement, more for analysis/simulation.

    # Position Management
    # int or None: Max bars to hold a position (None = no time limit).
    # This parameter is now a default, overridden by volatility_regime_max_holding_bars if enabled.
    "max_holding_period_bars": 25,

    "exit_on_neutral_signal": False,     # bool: If True, close position on neutral (0) signal. If False, ignore neutral signals while in trade.
    "allow_long_trades": True,          # bool: If True, allow opening long positions (signal 1).
    "allow_short_trades": True,         # bool: If True, allow opening short positions (signal -1).

    # --- Volatility Regime Filter Parameters (NEW) ---
    # bool: Enable filtering trades and adjusting max holding period based on volatility regime?
    "volatility_regime_filter_enabled": True,
    # dict[int, int | None]: Maps volatility regime (0=low, 1=medium, 2=high) to max holding period in bars.
    # Set to None for no time limit in that regime.
    "volatility_regime_max_holding_bars": {
        0: 5,   # Low Volatility: Shorter holding period
        1: 21,  # Medium Volatility: Moderate holding period
        2: 11   # High Volatility: Longer holding period (allow trends to run)
    },
    # dict[int, bool]: Maps volatility regime (0=low, 1=medium, 2=high) to whether trading is allowed.
    "allow_trading_in_volatility_regime": {
        0: True, # Low Volatility: Do not trade
        1: True,  # Medium Volatility: Allow trading
        2: False   # High Volatility: Allow trading
    },


    # Stop-Loss and Take-Profit Strategy (for Backtester/Live Bot Execution)
    # Note: These are separate from LABELING_CONFIG parameters
    'volatility_adjustment_enabled': True,# bool: Use ATR-based dynamic SL/TP for execution?
    'volatility_window_bars': 20,       # int: Lookback period for ATR calculation (if enabled). Should ideally match labeling config if used there.
    'fixed_take_profit_pct': 8.0,       # float (0-100): Fixed TP percentage for execution. MUST BE > 0.
    'fixed_stop_loss_pct': 2.0,         # float (0-100): Fixed SL percentage for execution. MUST BE > 0.
    'alpha_take_profit': 14.0,           # float: ATR multiplier for dynamic TP (if enabled).
    'alpha_stop_loss': 6,             # float: ATR multiplier for dynamic SL (if enabled).

    # Additional Filters
    'trend_filter_enabled': False,      # bool: Apply EMA trend filter to signals?
    "trend_filter_ema_period": 20,     # int: Period for the trend filter EMA (if enabled).

    # --- Confidence Filter Parameters (NEW) ---
    'confidence_filter_enabled': True,  # bool: Enable filtering trades based on model confidence?
    # float (0-100): Minimum probability/confidence required for a LONG signal to be considered valid for entry.
    'confidence_threshold_long_pct': 60.0, # Example: require at least 60% confidence for LONG
    # float (0-100): Minimum probability/confidence required for a SHORT signal to be considered valid for entry.
    'confidence_threshold_short_pct': 60.0, # Example: require at least 60% confidence for SHORT


    # Data Handling
    "data_lookback_bars": 300,          # int: Initial number of bars to fetch for the bot's buffer. Ensure this is > max(feature_lookback, sequence_length).
    'sequence_length_bars': 5,         # int: Sequence length for model input (MUST match FEATURE/MODEL config).

    # Live Bot Specifics
    "loop_interval_sec": None,          # int/float or None: Override candle interval wait time (None=wait for candle close).
    "exchange_type": "binance_futures", # str: Exchange adapter key (MUST match a key in EXCHANGE_CONFIG).

    # Notification Settings
    "notifier_params": {                # dict: Configuration passed to NotificationManager.
        "telegram": {
            "enabled": True,            # bool: Enable Telegram notifications?
            "token": TELEGRAM_BOT_TOKEN,     # str: Bot token (from env var).
            "chat_id": TELEGRAM_CHAT_ID,   # str: Chat ID (from env var).
            "level": "INFO",            # str: Min log level for notifications ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        },
        # Add other services like Slack here if needed
        # "slack": { ... },
    },
}

# --- Backtesting Specific Configuration ---
# Overrides or adds parameters specifically for the backtesting engine.
BACKTESTER_CONFIG = {
    # --- Overrides for Strategy Parameters (Optional) ---
    # Uncomment and modify to use different settings during backtests vs. live.
    'initial_capital': 1000.0,
    # 'leverage': 10,
    # 'risk_per_trade_pct': 1.0, # e.g., Risk 1% in backtests
    # 'max_holding_period_bars': 120, # This will be overridden by volatility_regime_max_holding_bars if enabled
    # 'exit_on_neutral_signal': False, # Example: Override for backtesting
    # 'allow_long_trades': False, # Example: Only allow short trades in backtest
    # 'allow_short_trades': False, # Example: Only allow long trades in backtest
    # 'volatility_adjustment_enabled': True,
    # 'fixed_take_profit_pct': 3.0,
    # 'fixed_stop_loss_pct': 1.0,
    # 'alpha_take_profit': 5.0,
    # 'alpha_stop_loss': 2.5,
    # 'trend_filter_enabled': True,
    # 'trend_filter_ema_period': 50,
    # 'confidence_filter_enabled': True, # Example: Enable confidence filter for backtesting
    # 'confidence_threshold_long_pct': 70.0, # Example: Use a higher threshold for backtesting
    # 'confidence_threshold_short_pct': 70.0,
    # 'volatility_regime_filter_enabled': True, # Example: Enable volatility filter for backtesting
    # 'volatility_regime_max_holding_bars': {0: 5, 1: 10, 2: 20}, # Example: Different holding periods for backtesting
    # 'allow_trading_in_volatility_regime': {0: False, 1: True, 2: True}, # Example: Same trading allowance for backtesting


    # --- Backtester Engine Parameters ---
    'maintenance_margin_rate': 0.005,   # float (fraction): Estimated maintenance margin rate for liquidation calculation (e.0.005 = 0.5%).
    'liquidation_fee_rate': 0.0005,     # float (fraction): Specific fee rate applied on simulated liquidation (defaults to trading_fee_rate if omitted).
    'max_concurrent_trades': 1,         # int: Max simultaneous trades (currently only 1 is fully supported).

    # --- Reporting Flags ---
    "save_trades": True,                # bool: Save detailed trade logs?
    "save_equity_curve": True,          # bool: Save equity curve data?
    "save_metrics": True,               # bool: Save summary performance metrics?
}

# --- Final Validation (Optional but Recommended) ---
# Add checks here to ensure consistency between different config sections if needed.
# Example: Ensure sequence lengths match across FEATURE, MODEL, and STRATEGY configs.
# Check if LSTM is available before accessing its config
if 'lstm' in MODEL_CONFIG and LSTM_AVAILABLE:
    if MODEL_CONFIG['lstm']['params'].get('sequence_length_bars') != FEATURE_CONFIG.get('sequence_length_bars') or \
       MODEL_CONFIG['lstm']['params'].get('sequence_length_bars') != STRATEGY_CONFIG.get('sequence_length_bars'):
        logger.warning("Sequence length mismatch between FEATURE_CONFIG, MODEL_CONFIG['lstm'], and STRATEGY_CONFIG!")
        logger.warning(f"  FEATURE_CONFIG['sequence_length_bars']: {FEATURE_CONFIG.get('sequence_length_bars')}")
        logger.warning(f"  MODEL_CONFIG['lstm']['params']['sequence_length_bars']: {MODEL_CONFIG['lstm']['params'].get('sequence_length_bars')}")
        logger.warning(f"  STRATEGY_CONFIG['sequence_length_bars']: {STRATEGY_CONFIG.get('sequence_length_bars')}")


# Check if required API keys are present if using Binance
if STRATEGY_CONFIG.get('exchange_type') == 'binance_futures':
    if not EXCHANGE_CONFIG.get('binance_futures', {}).get('api_key') or not EXCHANGE_CONFIG.get('binance_futures', {}).get('api_secret'):
         logger.critical("Binance API Key or Secret is missing but required for the selected exchange type.")
         # Consider exiting or raising an error in a real application startup
         # sys.exit(1)

# Check if required Telegram credentials are present if enabled
if STRATEGY_CONFIG.get('notifier_params', {}).get('telegram', {}).get('enabled'):
     if not STRATEGY_CONFIG.get('notifier_params', {}).get('telegram', {}).get('token') or not STRATEGY_CONFIG.get('notifier_params', {}).get('telegram', {}).get('chat_id'):
          logger.warning("Telegram notifications enabled, but Token or Chat ID is missing. Notifications will likely fail.")

# Check if volatility regime periods are valid for the filter
if STRATEGY_CONFIG.get('volatility_regime_filter_enabled', False):
    regime_max_holding = STRATEGY_CONFIG.get('volatility_regime_max_holding_bars')
    allow_trading = STRATEGY_CONFIG.get('allow_trading_in_volatility_regime')
    if not isinstance(regime_max_holding, dict) or set(regime_max_holding.keys()) != {0, 1, 2}:
        logger.critical("CRITICAL: 'volatility_regime_max_holding_bars' must be a dictionary with keys 0, 1, 2.")
        # Consider exiting or raising an error
    if not isinstance(allow_trading, dict) or set(allow_trading.keys()) != {0, 1, 2}:
         logger.critical("CRITICAL: 'allow_trading_in_volatility_regime' must be a dictionary with keys 0, 1, 2.")
         # Consider exiting or raising an error


logger.info("Parameters configuration loaded.")
