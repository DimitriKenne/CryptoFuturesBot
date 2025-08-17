#!/usr/bin/env python3
"""
generate_features.py

Loads raw OHLCV data using DataManager, engineers features using the FeatureEngineer,
and saves the resulting DataFrame using DataManager.

Uses the updated configuration structure from config/params.py and config/feature_config_schema.py.
Configures logging using utils/logger_config.py.
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
    # Import the central app_config object and FLOAT_EPSILON
    from config.params import app_config, FLOAT_EPSILON
    # Import the DataManager
    from utils.data_management.data_manager import DataManager
    # Import FeatureEngineer from its location
    from utils.feature_engineering.feature_engineer import FeatureEngineer
    # Import TemporalSafetyError
    from utils.exceptions import TemporalSafetyError
    # Import setup_rotating_logging
    from utils.logger_config import setup_rotating_logging
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import necessary modules: {e}. "
          f"Ensure your project structure and dependencies are correct.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during initial imports or configuration loading: {e}", file=sys.stderr)
    sys.exit(1)

# --- Set up Logging ---
try:
    setup_rotating_logging('generate_features')
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

def generate_features_pipeline(symbol: str, interval: str):
    """
    Main pipeline to load raw OHLCV data, engineer features, and save them.

    Args:
        symbol (str): Trading pair symbol (e.g., 'ADAUSDT').
        interval (str): Time interval for candles (e.g., '5m').
    """
    logger.info(f"Starting feature generation pipeline for {symbol} {interval}...")

    # --- 1. Load Feature Engineering Configuration from app_config ---
    # Use the features section from the central app_config
    feature_config = copy.deepcopy(app_config.features)
    # Corrected: Access sequence_length_bars and temporal_validation directly from feature_config
    # These attributes are part of FeatureConfig itself, not TradingConfig.
    # We no longer need a separate 'strategy_config' here for overrides.
    general_config = copy.deepcopy(app_config.general)


    logger.info(f"Final feature engineering configuration: {feature_config}")
    logger.info(f"Using general configuration: {general_config}")


    # --- 2. Initialize DataManager and FeatureEngineer ---
    dm = DataManager()
    try:
        fe = FeatureEngineer(config=feature_config)
    except Exception as e:
        logger.error(f"An unexpected error occurred initializing FeatureEngineer: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Load Raw Data ---
    logger.info(f"Attempting to load raw data for {symbol} {interval}...")
    try:
        df_raw = dm.load_data(
            symbol=symbol,
            interval=interval,
            data_type='raw'
        )
        logger.info(f"Successfully loaded raw data for {symbol} {interval}. Shape: {df_raw.shape}")
    except FileNotFoundError:
        logger.critical(f"Raw data file not found for {symbol} {interval}. "
                        f"Please run 'python -m scripts.fetch_data --symbol {symbol} --interval {interval}' first.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error loading raw data for {symbol} {interval}: {e}", exc_info=True)
        sys.exit(1)

    if df_raw.empty:
        logger.critical(f"Loaded raw data for {symbol} {interval} is empty. Cannot generate features.")
        sys.exit(1)
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        logger.critical("Loaded DataFrame does not have a DatetimeIndex. Please ensure your data fetching pipeline sets the index correctly.")
        sys.exit(1)
    logger.info("Raw data basic validation passed.")

    # --- 4. Generate Features ---
    try:
        logger.info("Generating features...")
        df_features = fe.process(df_raw)
        logger.info(f"Successfully generated features. Final DataFrame shape: {df_features.shape}")
    except TemporalSafetyError as e:
        logger.critical(f"Temporal safety error during feature generation: {e}", exc_info=True)
        logger.critical(f"Violating features: {e.features}. Please review feature engineering logic.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during feature generation: {e}", exc_info=True)
        sys.exit(1)

    # --- 5. Save Processed Data ---
    logger.info(f"Attempting to save processed (featured) data for {symbol} {interval}...")
    try:
        dm.save_data(
            df_to_save=df_features,
            symbol=symbol,
            interval=interval,
            data_type='processed',
        )
        logger.info(f"Successfully saved processed data to {dm.get_file_path(symbol, interval, 'processed')}")
    except Exception as e:
        logger.critical(f"Error saving processed data: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Feature generation pipeline for {symbol} {interval} completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate technical and statistical features from OHLCV data.'
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

    args = parser.parse_args()

    try:
        generate_features_pipeline(
            symbol=args.symbol,
            interval=args.interval,
        )
    except SystemExit:
        pass
    except Exception: 
        logger.error("Feature generation script terminated due to an unhandled error.", exc_info=True) 
        sys.exit(1)

    """
    Usage example:

    Generate features for BTCUSDT 1-hour data:
        python scripts/generate_features.py --symbol BTCUSDT --interval 1h

    Generate features for ADAUSDT 5-minute data:
        python -m scripts.generate_features --symbol ADAUSDT --interval 5m
    """
