# config/paths.py

"""
Centralized path configurations for the trading bot project.

Defines directories and file naming patterns for raw data, processed data,
labeled data, trained models, logs, and results.
Ensures consistency across various scripts and modules.
"""

from pathlib import Path

# --- Base Directories ---
# Assuming project root is the directory containing this 'config' folder
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"

# Ensure all base directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# --- Data Subdirectories ---
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELED_DATA_DIR = DATA_DIR / "labeled"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LABELED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Model Subdirectories ---
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Results Subdirectories ---
BACKTESTING_RESULTS_DIR = RESULTS_DIR / "backtesting"
LIVE_TRADING_RESULTS_DIR = RESULTS_DIR / "live_trading"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis" # Base analysis directory

BACKTESTING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LIVE_TRADING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Analysis subdirectories
MODEL_ANALYSIS_DIR = ANALYSIS_RESULTS_DIR / "model"
LABELING_ANALYSIS_BASE_DIR = ANALYSIS_RESULTS_DIR / "labeling" # Base directory for all labeling analysis
BACKTESTING_ANALYSIS_DIR = ANALYSIS_RESULTS_DIR / "backtesting"
LIVE_TRADING_ANALYSIS_DIR = ANALYSIS_RESULTS_DIR / "live_trading"

MODEL_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
LABELING_ANALYSIS_BASE_DIR.mkdir(parents=True, exist_ok=True)
BACKTESTING_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
LIVE_TRADING_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# --- File Naming Patterns ---
RAW_DATA_PATTERN = "{symbol}_{interval}_raw.csv"
PROCESSED_DATA_PATTERN = "{symbol}_{interval}_processed.parquet"
LABELED_DATA_PATTERN = "{symbol}_{interval}_labeled.parquet" # Pattern for labeled data files

TRAINED_MODEL_PATTERN = "{symbol}_{interval}_{model_key}_model.keras" # For Keras models
MODEL_METADATA_PATTERN = "{symbol}_{interval}_{model_key}_metadata.json"
MODEL_PREPROCESSOR_PATTERN = "{symbol}_{interval}_{model_key}_preprocessor.pkl"
MODEL_PIPELINE_PATTERN = "{symbol}_{interval}_{model_key}_pipeline.pkl" # For sklearn pipelines

# Backtesting Results Patterns
# These should now include {model_type} and use .parquet
BACKTESTING_TRADES_PATTERN = "{symbol}_{interval}_{model_type}_trades.parquet"
BACKTESTING_EQUITY_PATTERN = "{symbol}_{interval}_{model_type}_equity.parquet"
BACKTESTING_METRICS_PATTERN = "{symbol}_{interval}_{model_type}_metrics.json"

# Live Trading Results Patterns (assuming similar structure)
# Changed to .parquet for consistency and performance
LIVE_TRADING_TRADES_PATTERN = "{symbol}_{interval}_{model_type}_trades.parquet"
LIVE_TRADING_EQUITY_PATTERN = "{symbol}_{interval}_{model_type}_equity.parquet"
LIVE_TRADING_METRICS_PATTERN = "{symbol}_{interval}_{model_type}_metrics.json"
LIVE_TRADING_CAPITAL_STATE_PATTERN = "{symbol}_{interval}_{model_type}_capital_state.json" # For bot state

# Analysis Output Patterns (within analysis subdirectories)
LABELING_STRATEGY_ANALYSIS_DIR_PATTERN_STR = str(LABELING_ANALYSIS_BASE_DIR / "{label_strategy}") # New pattern for strategy-specific subfolders
LABELING_ANALYSIS_PLOT_PATTERN = "{symbol}_{interval}_{analysis_type}.png" # Updated to remove strategy from filename
LABELING_ANALYSIS_TABLE_PATTERN = "{symbol}_{interval}_{analysis_type}.csv" # Updated to remove strategy from filename

ANALYSIS_PLOT_PATTERN = "{symbol}_{interval}_{model_type}_{analysis_type}.png" # Generic plot pattern for model/backtest analysis
ANALYSIS_TABLE_PATTERN = "{symbol}_{interval}_{model_type}_{analysis_type}.csv" # Generic table pattern for model/backtest analysis


# Consolidated dictionary of all paths for easy access
PATHS = {
    "project_root": PROJECT_ROOT,
    "data_dir": DATA_DIR,
    "models_dir": MODELS_DIR,
    "results_dir": RESULTS_DIR,
    "logs_dir": LOGS_DIR,
    "docs_dir": DOCS_DIR,

    "raw_data_dir": RAW_DATA_DIR,
    "processed_data_dir": PROCESSED_DATA_DIR,
    "labeled_data_dir": LABELED_DATA_DIR, # Base directory for labeled data files

    "trained_models_dir": TRAINED_MODELS_DIR,
    "backtesting_results_dir": BACKTESTING_RESULTS_DIR,
    "live_trading_results_dir": LIVE_TRADING_RESULTS_DIR,
    "analysis_dir": ANALYSIS_RESULTS_DIR, # Base analysis directory
    "model_analysis_dir": MODEL_ANALYSIS_DIR,
    "labeling_analysis_base_dir": LABELING_ANALYSIS_BASE_DIR, # Base directory for all labeling analysis
    "labeling_strategy_analysis_dir_pattern": LABELING_STRATEGY_ANALYSIS_DIR_PATTERN_STR, # NEW: now a string pattern
    "backtesting_analysis_dir": BACKTESTING_ANALYSIS_DIR,
    "live_trading_analysis_dir": LIVE_TRADING_ANALYSIS_DIR,

    "raw_data_pattern": RAW_DATA_PATTERN,
    "processed_data_pattern": PROCESSED_DATA_PATTERN,
    "labeled_data_pattern": LABELED_DATA_PATTERN, # Pattern for labeled data files

    "trained_model_pattern": TRAINED_MODEL_PATTERN,
    "model_metadata_pattern": MODEL_METADATA_PATTERN,
    "model_preprocessor_pattern": MODEL_PREPROCESSOR_PATTERN,
    "model_pipeline_pattern": MODEL_PIPELINE_PATTERN,

    "backtesting_trades_pattern": BACKTESTING_TRADES_PATTERN,
    "backtesting_equity_pattern": BACKTESTING_EQUITY_PATTERN,
    "backtesting_metrics_pattern": BACKTESTING_METRICS_PATTERN,

    "live_trading_trades_pattern": LIVE_TRADING_TRADES_PATTERN,
    "live_trading_equity_pattern": LIVE_TRADING_EQUITY_PATTERN,
    "live_trading_metrics_pattern": LIVE_TRADING_METRICS_PATTERN,
    "live_trading_capital_state_pattern": LIVE_TRADING_CAPITAL_STATE_PATTERN,

    "labeling_analysis_plot_pattern": LABELING_ANALYSIS_PLOT_PATTERN,
    "labeling_analysis_table_pattern": LABELING_ANALYSIS_TABLE_PATTERN,

    "analysis_plot_pattern": ANALYSIS_PLOT_PATTERN, # Generic for model/backtest analysis
    "analysis_table_pattern": ANALYSIS_TABLE_PATTERN, # Generic for model/backtest analysis
}
