#!/usr/bin/env python3
"""
Script to perform key analyses on PRE-EXISTING generated trading labels.

This script validates the application configuration using the central validator
and re-runs common analyses on data that has already been labeled.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from typing import List
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import configuration and refactored components ---
try:
    from config.params import app_config
    from config.validator import validate_config
    from utils.data_management.data_manager import DataManager
    from utils.labeling.label_analyzer import LabelAnalyzer
    from utils.labeling.analysis_plotter import AnalysisPlotter
    from utils.labeling.analysis_calculator import AnalysisCalculator
    from utils.labeling.label_generator import LABELING_STRATEGY_MAP
    from utils.logger_config import setup_rotating_logging
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import modules. Check project setup. Error: {e}", file=sys.stderr)
    sys.exit(1)

# Set up Logging
setup_rotating_logging('label_analysis')
logger = logging.getLogger(__name__)

def analyze_labels_pipeline(symbol: str, interval: str, labeling_strategy_name: str, future_horizons: List[int]):
    """
    Loads existing data and runs the common analysis suite.
    """
    logger.info(f"--- Starting Label Analysis for {symbol} {interval} with strategy '{labeling_strategy_name}' ---")

    # --- 0. CONFIGURATION VALIDATION ---
    logger.info("Validating labeling configuration...")
    try:
        validate_config(app_config.labeling)
        logger.info("Configuration validated successfully.")
    except (ValueError, TypeError) as e:
        logger.critical(f"Labeling configuration is invalid: {e}", exc_info=True)
        sys.exit(1)

    # --- 1. INITIALIZATION PHASE ---
    logger.info("Initializing components...")
    try:
        dm = DataManager()
        plotter = AnalysisPlotter(logger=logger)
        
        fee_rate = app_config.labeling.trading_fee_pct / 100.0
        slippage_rate = app_config.labeling.slippage_tolerance_pct / 100.0
        calculator = AnalysisCalculator(trading_fee_rate=fee_rate, slippage_tolerance_rate=slippage_rate)
        
        analyzer = LabelAnalyzer(
            data_manager=dm,
            calculator=calculator,
            plotter=plotter,
            logger=logger
        )
        logger.info("Components initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. EXECUTION PHASE ---
    logger.info(f"Loading data for {symbol} {interval}...")
    df_processed = dm.load_dataframe(data_type='processed', symbol=symbol, interval=interval)
    df_labeled = dm.load_dataframe(data_type='labeled', symbol=symbol, interval=interval)

    if df_processed is None or df_labeled is None:
        logger.critical("Processed or Labeled data not found. Please run create_labels.py first. Exiting.")
        sys.exit(1)
    
    logger.info("Combining processed and labeled data...")
    df_combined = df_processed.join(df_labeled, how='inner')
    df_combined['label'] = df_combined['label'].fillna(0).astype(int)
    logger.info(f"Combined data shape: {df_combined.shape}")

    strategy_config = getattr(app_config.labeling, labeling_strategy_name, None)
    f_window = getattr(strategy_config, "future_return_window", 150) if strategy_config else 150
    resolved_horizons = future_horizons or app_config.labeling.analysis_future_horizons or [10, 30, 60]
    
    # *** THIS IS THE FIX ***
    # Derive the clean strategy class name for consistent directory naming.
    try:
        StrategyClass = LABELING_STRATEGY_MAP[labeling_strategy_name]
        clean_strategy_name = StrategyClass.__name__
        logger.info(f"Resolved strategy key '{labeling_strategy_name}' to clean name '{clean_strategy_name}' for analysis.")
    except KeyError:
        logger.critical(f"Strategy key '{labeling_strategy_name}' not found in LABELING_STRATEGY_MAP.")
        sys.exit(1)

    logger.info("Performing common analyses...")
    analyzer.perform_all_analyses(
        df_combined=df_combined,
        symbol=symbol,
        interval=interval,
        labeling_strategy=clean_strategy_name,  # Use the clean name
        future_horizons=resolved_horizons,
        f_window=f_window
    )

    logger.info(f"--- Label analysis for {symbol} {interval} completed successfully! ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform analyses on existing trading labels.')
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, required=True, help='Time interval (e.g., 5m, 1h)')
    parser.add_argument('--labeling-strategy', type=str, required=True, choices=LABELING_STRATEGY_MAP.keys(), help='Labeling strategy name used to create the labels.')
    parser.add_argument('--future-horizons', type=int, nargs='*', default=[], help='Optional list of future horizons to analyze.')
    args = parser.parse_args()

    try:
        analyze_labels_pipeline(
            symbol=args.symbol,
            interval=args.interval,
            labeling_strategy_name=args.labeling_strategy,
            future_horizons=args.future_horizons
        )
    except SystemExit:
        pass
    except Exception as e:
        logger.exception("Label analysis script terminated due to an unhandled error.")
        sys.exit(1)