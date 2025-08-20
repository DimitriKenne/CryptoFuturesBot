#!/usr/bin/env python3
"""
Script to perform key analyses on generated trading labels and processed data.

Analyses performed:
1. Label Distribution
2. Maximum Favorable Excursion (MFE) / Maximum Adverse Excursion (MAE)
3. Future Returns
4. Profitability by Volatility Regime

Requires labeled data (output of create_labels.py) and corresponding processed
data, which must include OHLCV and 'volatility_regime' features.
Uses configuration from config/paths.py, config/params.py, and config/label_config_schema.py.
Configures logging using utils/logger_config.py.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration and utilities
try:
    from config.paths import PATHS
    from config.params import app_config
    from utils.data_management.data_manager import DataManager
    from utils.labeling.label_analyzer import LabelAnalyzer
    from utils.logger_config import setup_rotating_logging
    from utils.labeling.label_generator import LabelGenerator # For argparse choices
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import modules. Check project setup. Error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during imports/config loading: {e}", file=sys.stderr)
    sys.exit(1)

# Set up Logging
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

def analyze_labels_pipeline(symbol: str, interval: str, labeling_strategy: str, future_horizons: List[int]):
    """
    Loads processed and labeled data, then performs various analyses.

    Args:
        symbol (str): Trading pair (e.g., 'ADAUSDT').
        interval (str): Time interval (e.g., '5m').
        labeling_strategy (str): Name of the labeling strategy that generated the labels.
        future_horizons (List[int]): List of future bars for return analysis.
    """
    logger.info(f"Starting label analysis for {symbol} {interval} with strategy '{labeling_strategy}'...")

    dm = DataManager()

    # Resolve future horizons
    resolved_future_horizons = future_horizons
    if not resolved_future_horizons and hasattr(app_config.labeling, 'analysis_future_horizons'):
        resolved_future_horizons = app_config.labeling.analysis_future_horizons
    if not resolved_future_horizons:
        resolved_future_horizons = [10, 30, 60]
        logger.warning("No 'analysis_future_horizons' in config.labeling. Using default [10, 30, 60].")

    logger.info(f"Future horizons for analysis: {resolved_future_horizons}")

    # Load Data
    logger.info(f"Loading processed data for {symbol} {interval}...")
    try:
        df_processed = dm.load_data(symbol=symbol, interval=interval, data_type='processed')
        if df_processed is None or df_processed.empty:
            raise FileNotFoundError(f"Processed data not found/empty for {symbol} {interval}.")
        logger.info(f"Loaded processed data. Shape: {df_processed.shape}")
    except Exception as e:
        logger.error(f"Error loading processed data: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Loading labeled data for {symbol} {interval} with strategy '{labeling_strategy}'...")
    try:
        df_labeled = dm.load_data(symbol=symbol, interval=interval, data_type='labeled')
        if df_labeled is None or df_labeled.empty:
            raise FileNotFoundError(f"Labeled data not found/empty for {symbol} {interval}.")
        logger.info(f"Loaded labeled data. Shape: {df_labeled.shape}")
    except Exception as e:
        logger.error(f"Error loading labeled data: {e}", exc_info=True)
        sys.exit(1)

    # Combine DataFrames
    logger.info("Combining processed and labeled data...")
    try:
        if 'label' not in df_labeled.columns:
            raise ValueError("Labeled DataFrame must contain a 'label' column.")

        df_labeled_aligned = df_labeled.reindex(df_processed.index)
        df_labeled_aligned['label'] = pd.to_numeric(df_labeled_aligned['label'], errors='coerce')

        df_combined = pd.merge(
            df_processed,
            df_labeled_aligned[['label']],
            left_index=True,
            right_index=True,
            how='left'
        )

        initial_label_nans = df_combined['label'].isna().sum()
        if initial_label_nans > 0:
            logger.warning(f"Found {initial_label_nans} NaN labels after combining. Filling with 0 (neutral).")
            df_combined['label'] = df_combined['label'].fillna(0).astype(pd.Int8Dtype())
        else:
            df_combined['label'] = df_combined['label'].astype(pd.Int8Dtype())

        if df_combined.empty:
            raise ValueError("Combined DataFrame empty after merging. Check data integrity.")
        logger.info(f"Combined data. Shape: {df_combined.shape}")
    except Exception as e:
        logger.error(f"Error during data combination: {e}", exc_info=True)
        sys.exit(1)

    # Initialize LabelAnalyzer and Perform Analyses
    logger.info("Initializing LabelAnalyzer and performing common analyses...")
    try:
        analyzer = LabelAnalyzer(
            paths=PATHS,
            logger=logger,
            fee=None,
            slippage=None,
            f_window=None,
            labeling_strategy_type=labeling_strategy
        )
        analyzer.perform_all_analyses(
            df_combined=df_combined,
            symbol=symbol,
            interval=interval,
            labeling_strategy=labeling_strategy,
            future_horizons=resolved_future_horizons
        )
        logger.info("All common analyses completed.")
    except Exception as e:
        logger.error(f"Error during common analysis execution: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Label analysis pipeline for {symbol} {interval} completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Perform analyses on generated trading labels and processed data.'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading pair (e.g., BTCUSDT)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Time interval (e.g., 5m, 1h, 1d)'
    )
    parser.add_argument(
        '--labeling-strategy',
        type=str,
        required=True,
        choices=LabelGenerator.get_available_labeling_strategies(),
        help=f'Labeling strategy used. Used for organizing analysis results. Available: {", ".join(LabelGenerator.get_available_labeling_strategies())}'
    )
    parser.add_argument(
        '--future-horizons',
        type=int,
        nargs='*',
        default=[],
        help='List of future bars (integers) to analyze returns over. E.g., --future-horizons 10 30 60. Defaults to config if not provided.'
    )

    args = parser.parse_args()

    try:
        analyze_labels_pipeline(
            symbol=args.symbol,
            interval=args.interval,
            labeling_strategy=args.labeling_strategy,
            future_horizons=args.future_horizons
        )
    except SystemExit:
        pass
    except Exception:
        logger.exception("Label analysis script terminated due to an unhandled error.")
        sys.exit(1)

    """
    Usage example:

    Run analyses for ADAUSDT 5m data with default future horizons:
        python scripts/analyze_labels.py --symbol ADAUSDT --interval 5m --labeling-strategy labeling_strategy_2

    Run analyses for specific horizons (10, 30, 60 bars):
        python scripts/analyze_labels.py --symbol BTCUSDT --interval 1h --labeling-strategy labeling_strategy_1 --future-horizons 10 30 60

    Ensure processed and labeled data files exist in your data/ directory.
    """
