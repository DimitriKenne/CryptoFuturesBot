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

# ==============================================================================
# --- NEW: Centralized Path Configuration Object (The "Blueprint") ---
# This is the new, preferred way for DataManager to get path info.
# We will populate this incrementally as we refactor each part of the project.
# ==============================================================================
PATH_CONFIG = {
    'directories': {
        # --- Base Directories ---
        'models_base': MODELS_DIR,
        'results_base': RESULTS_DIR,

        # --- Data Subdirectories ---
        'raw': DATA_DIR / "raw",
        'processed': DATA_DIR / "processed",
        'labeled': DATA_DIR / "labeled",

        # --- Results Subdirectories ---
        'labeling': RESULTS_DIR / "labeling",
        'model_analysis': RESULTS_DIR / "model_analysis",
        'backtesting': RESULTS_DIR / "backtesting",
        'live_trading': RESULTS_DIR / "live_trading",
        'monte_carlo': RESULTS_DIR / "monte_carlo",
    },
    'patterns': {
        # --- Data File Patterns ---
        'raw_data': "{symbol}_{interval}_raw.parquet",
        'processed_data': "{symbol}_{interval}_processed.parquet",
        'labeled_data': "{symbol}_{interval}_labeled.parquet", # CORRECTED: The official, single labeled file

        # --- Labeling Analysis Directory & File Patterns ---
        'labeling_run_dir': "{symbol}_{interval}",
        'labeling_plot': "{analysis_type}_{labeling_strategy}.png", # Plots are distinguished by strategy
        'labeling_table': "{analysis_type}_{labeling_strategy}.csv", 


        # --- Model Training Directory & File Patterns ---
        'model_run_dir': "{model_type}/{symbol}_{interval}",
        'model_pipeline': "pipeline.pkl",
        'model_metadata': "metadata.json",
        'model_preprocessor': "preprocessor.pkl",
        'model_keras': "model.keras",
        'model_sampler': "sampler.pkl",
        'model_evaluation': "evaluation_metrics.json",
        'model_feature_importance': "feature_importance.csv",
        'model_plot': "{plot_type}.png",

        # --- Backtesting Directory & File Patterns ---
        'backtesting_run_dir': "{model_type}/{symbol}_{interval}",
        'backtest_trades': "trades.parquet",
        'backtest_equity': "equity.parquet",
        'backtest_metrics_json': "summary_metrics.json",
        'backtest_metrics_csv': "summary_metrics.csv",
        'backtest_plot': "{plot_type}.png",

        # --- Monte Carlo Directory & File Patterns ---
        'monte_carlo_run_dir': "{model_type}/{symbol}_{interval}/mode_{mode}_sims_{num_simulations}",
        'mc_summary_stats': "1_performance_summary_stats.csv",
        'mc_raw_metrics': "all_simulation_metrics.csv",
        'mc_plot': "{plot_type}.png",
        
        # --- Live Trading Directory & File Patterns ---
        'live_trading_run_dir': "{model_type}/{symbol}_{interval}",
        'live_trades': "trades.parquet",
        'live_equity': "equity.parquet",
        'live_metrics_json': "summary_metrics.json",
        'live_trading_state': "bot_state.db", # NEW: For saving bot state
        
        # --- Visualization File Pattern ---
        'trade_visualization_json': "trade_visualization_data.json",
    }
}
