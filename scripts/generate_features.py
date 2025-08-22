#!/usr/bin/env python3
"""
generate_features.py

Loads raw OHLCV data using DataManager, engineers features using the FeatureEngineer,
and saves the resulting DataFrame using DataManager. This script has been updated
to use the new config-driven methods in DataManager.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import copy
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration and utilities
try:
    from config.params import app_config
    from config.validator import validate_config  # <-- ADDED for robustness
    from utils.data_management.data_manager import DataManager
    from utils.feature_engineering.feature_engineer import FeatureEngineer
    from utils.exceptions import TemporalSafetyError
    from utils.logger_config import setup_rotating_logging
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import necessary modules: {e}.", file=sys.stderr)
    sys.exit(1)

# --- Set up Logging ---
try:
    setup_rotating_logging('generate_features')
    logger = logging.getLogger(__name__)
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to configure rotating logging: {e}. Using basic stdout logging.", exc_info=True)

def generate_features_pipeline(symbol: str, interval: str):
    """
    Main pipeline to load raw OHLCV data, engineer features, and save them.

    Args:
        symbol (str): Trading pair symbol (e.g., 'ADAUSDT').
        interval (str): Time interval for candles (e.g., '5m').
    """
    logger.info(f"Starting feature generation pipeline for {symbol} {interval}...")

    # --- 1. Load and Validate Feature Configuration ---
    feature_config = copy.deepcopy(app_config.features)
    logger.info("Validating feature configuration...")
    try:
        validate_config(feature_config)
        logger.info("Feature configuration is valid.")
    except Exception as e:
        logger.critical(f"Feature configuration is invalid: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Initialize DataManager and FeatureEngineer ---
    dm = DataManager()
    fe = FeatureEngineer(config=feature_config)

    # --- 3. Load Raw Data using the new DataManager method ---
    logger.info(f"Attempting to load raw data for {symbol} {interval}...")
    try:
        # UPDATED: Use the new, config-driven load_dataframe method
        df_raw = dm.load_dataframe(
            data_type='raw',
            symbol=symbol,
            interval=interval
        )
        if df_raw is None or df_raw.empty:
            logger.critical(f"Raw data file not found or is empty for {symbol} {interval}. "
                            f"Please run 'python scripts/fetch_data.py --symbol {symbol} --interval {interval}' first.")
            sys.exit(1)
        
        logger.info(f"Successfully loaded raw data. Shape: {df_raw.shape}")

    except Exception as e:
        logger.critical(f"Error loading raw data for {symbol} {interval}: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Generate Features ---
    try:
        logger.info("Generating features...")
        df_features = fe.process(df_raw)
        logger.info(f"Successfully generated features. Final DataFrame shape: {df_features.shape}")
    except TemporalSafetyError as e:
        logger.critical(f"Temporal safety error during feature generation: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during feature generation: {e}", exc_info=True)
        sys.exit(1)

    # --- 5. Save Processed Data using the new DataManager method ---
    logger.info(f"Attempting to save processed (featured) data for {symbol} {interval}...")
    try:
        # UPDATED: Use the new, config-driven save_dataframe method.
        # This method handles its own logging, making this call cleaner.
        dm.save_dataframe(
            df=df_features,
            data_type='processed',
            symbol=symbol,
            interval=interval,
        )
    except Exception as e:
        logger.critical(f"Error saving processed data: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Feature generation pipeline for {symbol} {interval} completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate technical and statistical features from OHLCV data.'
    )
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument(
        '--interval', type=str, required=True,
        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Time interval for candles (e.g., 5m, 1h, 1d)'
    )

    args = parser.parse_args()

    try:
        generate_features_pipeline(symbol=args.symbol, interval=args.interval)
    except SystemExit:
        pass # Allow clean exit
    except Exception: 
        logger.error("Feature generation script terminated due to an unhandled error.", exc_info=True) 
        sys.exit(1)