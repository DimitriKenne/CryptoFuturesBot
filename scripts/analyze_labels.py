#!/usr/bin/env python3
"""
Script to perform key analyses on generated trading labels and processed data
to inform strategy configuration (stop-loss, take-profit, max_holding_period).

Analyses performed:
1.  Label Streak Analysis (Signal Duration)
2.  Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE) Analysis
3.  Future Returns Analysis (Post-Signal)
4.  Profitability Analysis by Volatility Regime

Requires labeled data (output of create_labels.py) and corresponding processed
data (input to create_labels.py) which must contain OHLCV data and features
like 'volatility_regime'.

Uses configuration from config/paths.py, config/params.py, and config/label_config_schema.py.
Configures logging using utils/logger_config.py (assumed to exist).
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv # Import load_dotenv
import copy # Import copy for deepcopy

# --- Load Environment Variables ---
load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration and utilities
try:
    from config.paths import PATHS
    # Import StrategyConfig and its DEFAULT_STRATEGY_CONFIG instance
    from config.strategy_config_schema import StrategyConfig, DEFAULT_STRATEGY_CONFIG
    # Import LabelConfig and its DEFAULT_LABEL_CONFIG instance
    from config.label_config_schema import LabelConfig, DEFAULT_LABEL_CONFIG, Strategy1Config, Strategy2Config, Strategy3Config, Strategy4Config

    from utils.data_manager import DataManager
    from utils.labeling.label_analyzer import LabelAnalyzer # Import LabelAnalyzer
    from utils.logger_config import setup_rotating_logging
    from utils.labeling.label_generator import LabelGenerator # Import LabelGenerator to get available strategies

except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import necessary modules. "
          f"Ensure your project structure and dependencies are correct. Error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during initial imports or configuration loading: {e}", file=sys.stderr)
    sys.exit(1)

# --- Set up Logging ---
try:
    setup_rotating_logging('label_analysis')
    logger = logging.getLogger(__name__)
    logger.info("Rotating logging configured successfully.")
except Exception as e:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(lineno)d]',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to configure rotating logging: {e}. Using basic stdout logging.", exc_info=True)


def analyze_labels_pipeline(symbol: str, interval: str, label_strategy: str, future_horizons: List[int]):
    """
    Main pipeline to load processed and labeled data, then perform various analyses.

    Args:
        symbol (str): Trading pair symbol (e.g., 'ADAUSDT').
        interval (str): Time interval for candles (e.g., '5m').
        label_strategy (str): The name of the labeling strategy that generated the labels.
                              Used for organizing analysis results.
        future_horizons (List[int]): List of future bars to analyze returns over.
    """
    logger.info(f"Starting label analysis script for {symbol} {interval} with strategy '{label_strategy}'...")

    dm = DataManager()

    # --- 1. Load Configuration ---
    # Start with a deep copy of the default LabelConfig dataclass instance
    label_config = copy.deepcopy(DEFAULT_LABEL_CONFIG)
    # Start with a deep copy of the default StrategyConfig dataclass instance (to get analysis horizons)
    strategy_config = copy.deepcopy(DEFAULT_STRATEGY_CONFIG)
    
    # Apply command-line override for label_type (for analysis naming consistency)
    label_config.label_type = label_strategy

    # Apply common parameters from StrategyConfig to label_config
    label_config.trading_fee_rate = strategy_config.trading_fee_rate
    label_config.slippage_tolerance_pct = strategy_config.slippage_tolerance_pct

    logger.info(f"Label analysis will use configuration: {label_config}")
    logger.info(f"Label analysis will use strategy configuration (for analysis horizons): {strategy_config}")


    # --- 2. Load Data ---
    logger.info(f"Loading processed data for {symbol} {interval}...")
    try:
        df_processed = dm.load_data(symbol=symbol, interval=interval, data_type='processed')
        if df_processed is None or df_processed.empty:
            raise FileNotFoundError(f"Processed data not found or is empty for {symbol} {interval}.")
        logger.info(f"Successfully loaded processed data. Shape: {df_processed.shape}")
    except Exception as e:
        logger.error(f"An error occurred during processed data loading: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Loading labeled data for {symbol} {interval} with strategy '{label_strategy}'...")
    try:
        df_labeled = dm.load_data(symbol=symbol, interval=interval, data_type='labeled')
        if df_labeled is None or df_labeled.empty:
            raise FileNotFoundError(f"Labeled data not found or is empty for {symbol} {interval}.")
        logger.info(f"Successfully loaded labeled data. Shape: {df_labeled.shape}")
    except Exception as e:
        logger.error(f"An error occurred during labeled data loading: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Combine DataFrames ---
    logger.info("Combining processed and labeled data...")
    try:
        if 'label' not in df_labeled.columns:
            raise ValueError("Labeled DataFrame must contain a 'label' column.")

        df_labeled_aligned = df_labeled.reindex(df_processed.index)
        df_labeled_aligned['label'] = df_labeled_aligned['label'].astype(pd.Int8Dtype())

        df_combined = pd.merge(
            df_processed,
            df_labeled_aligned[['label']],
            left_index=True,
            right_index=True,
            how='left'
        )

        initial_label_nans = df_combined['label'].isna().sum()
        if initial_label_nans > 0:
            logger.warning(f"Found {initial_label_nans} NaN values in 'label' column after combining. Filling with 0 (neutral).")
            df_combined['label'] = df_combined['label'].fillna(0).astype(pd.Int8Dtype())

        if df_combined.empty:
            raise ValueError("Combined DataFrame is empty after merging. Check data integrity.")
        logger.info(f"Successfully combined data. Shape: {df_combined.shape}")

    except Exception as e:
        logger.error(f"An error occurred during data combination: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Initialize LabelAnalyzer and Perform Analyses ---
    logger.info("Initializing LabelAnalyzer...")
    try:
        # Use values directly from the configured label_config instance for LabelAnalyzer
        fee_param = label_config.trading_fee_rate
        slippage_param = label_config.slippage_tolerance_pct
        
        # Get f_window from the specific strategy config within label_config
        strategy_config_obj = getattr(label_config, label_config.label_type)
        f_window_param = getattr(strategy_config_obj, 'future_return_window', 150) # Use a reasonable default

        analyzer = LabelAnalyzer(paths=PATHS, logger=logger, fee=fee_param, slippage=slippage_param, f_window=f_window_param)
        logger.info(f"Performing all analyses for {symbol} {interval} with strategy '{label_strategy}'...")

        analyzer.perform_all_analyses(
            df_combined=df_combined,
            symbol=symbol,
            interval=interval,
            label_strategy=label_strategy,
            future_horizons=future_horizons
        )
        logger.info("All selected analyses completed.")

    except Exception as e:
        logger.error(f"An error occurred during analysis execution: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Label analysis pipeline for {symbol} {interval} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform various analyses on generated trading labels and processed data.'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading pair symbol (e.g., BTCUSDT)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Time interval for candles (e.g., 5m, 1h, 1d)'
    )
    parser.add_argument(
        '--label-strategy',
        type=str,
        required=True,
        choices=LabelGenerator.get_available_strategies(), # Dynamically get choices
        help=f'Labeling strategy used to generate the labels. This is used for organizing analysis results. Available: {", ".join(LabelGenerator.get_available_strategies())}'
    )
    parser.add_argument(
        '--future-horizons',
        type=int,
        nargs='*', # 0 or more arguments
        default=DEFAULT_STRATEGY_CONFIG.analysis_future_horizons, # Default from StrategyConfig
        help='List of future bars (integers) to analyze returns over. E.g., --future-horizons 10 30 60. Defaults to config setting.'
    )

    args = parser.parse_args()

    try:
        analyze_labels_pipeline(
            symbol=args.symbol,
            interval=args.interval,
            label_strategy=args.label_strategy,
            future_horizons=args.future_horizons
        )
    except SystemExit:
        pass # Prevent traceback on intentional sys.exit() calls
    except Exception:
        logger.exception("Label analysis script terminated due to an unhandled error.")
        sys.exit(1)

    """
    Usage example:

    Run all analyses for ADAUSDT 5m data with default future horizons:
        python scripts/analyze_labels.py --symbol ADAUSDT --interval 5m --label-strategy strategy_2

    Run analyses for specific horizons (10, 30, 60 bars):
        python scripts/analyze_labels.py --symbol BTCUSDT --interval 1h --label-strategy strategy_1 --future-horizons 10 30 60

    Ensure you have processed data files and a labeled data file (e.g., ADAUSDT_5m_labeled.parquet)
    in your data/ and that config/params.py and config/paths.py are correct.
    """
