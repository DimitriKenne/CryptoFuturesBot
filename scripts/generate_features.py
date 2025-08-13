#!/usr/bin/env python3
"""
generate_features.py

Loads raw OHLCV data using DataManager, engineers features using the FeatureEngineer,
and saves the resulting DataFrame using DataManager.

Uses the updated configuration structure from config/params.py and config/paths.py.
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
    # Import DEFAULT_FEATURE_CONFIG from the new schema file in config
    # We no longer need to import TemporalValidationConfig directly here for instantiation,
    # as FeatureConfig will handle its internal instantiation.
    from config.feature_config_schema import DEFAULT_FEATURE_CONFIG # Only import DEFAULT_FEATURE_CONFIG

    # Import STRATEGY_CONFIG from params.py (as it's remaining for now)
    from config.params import STRATEGY_CONFIG

    # Import centralized path configurations
    from config.paths import PATHS

    # Import the DataManager
    from utils.data_manager import DataManager
    # Import FeatureEngineer from its new location
    from utils.feature_engineering.feature_engineer import FeatureEngineer
    # Assuming TemporalSafetyError is defined in a custom exceptions.py file
    from utils.exceptions import TemporalSafetyError

    # Import setup_rotating_logging
    from utils.logger_config import setup_rotating_logging

except ImportError as e:
    print(f"ERROR: Failed to import necessary modules. Ensure config/, utils/, and exceptions.py are correctly structured. Error: {e}")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"ERROR: Configuration file not found: {e}. Ensure config/params.py, config/paths.py, and exceptions.py exist.")
    sys.exit(1)
except AttributeError as e:
     print(f"ERROR: Configuration object missing expected attribute or key: {e}. Check config/params.py and config/paths.py.")
     sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during initial imports or configuration loading: {e}")
    sys.exit(1)


# --- Configure Rotating Logging ---
setup_rotating_logging('generate_features', logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Rotating logging configured successfully.")


# --- Feature Generation Logic ---
def generate_features_pipeline(symbol: str, interval: str):
    """
    End-to-end feature generation pipeline: loads raw data using DataManager,
    engineers features using the FeatureEngineer, and saves the resulting
    processed data using DataManager. Handles errors including temporal safety violations.
    """
    logger.info(f"Starting feature generation pipeline for {symbol.upper()} {interval}...")

    dm = DataManager()

    # --- Prepare configuration for FeatureEngineer ---
    # Start with a deep copy of DEFAULT_FEATURE_CONFIG from feature_config_schema.py
    engineer_init_config = copy.deepcopy(DEFAULT_FEATURE_CONFIG)

    # Handle temporal_validation override from STRATEGY_CONFIG
    # IMPORTANT: Only update the dictionary. Do NOT manually instantiate TemporalValidationConfig here.
    # FeatureConfig(**engineer_init_config) will handle the nested instantiation.
    if 'temporal_validation' in STRATEGY_CONFIG and isinstance(STRATEGY_CONFIG['temporal_validation'], dict):
        # Ensure 'temporal_validation' exists as a dictionary in engineer_init_config
        # before trying to update it, in case DEFAULT_FEATURE_CONFIG didn't have it (though it should)
        if 'temporal_validation' not in engineer_init_config or not isinstance(engineer_init_config['temporal_validation'], dict):
             engineer_init_config['temporal_validation'] = {}
        
        engineer_init_config['temporal_validation'].update(STRATEGY_CONFIG['temporal_validation'])
    
    # Handle sequence_length_bars override from STRATEGY_CONFIG
    if 'sequence_length_bars' in STRATEGY_CONFIG:
        engineer_init_config['sequence_length_bars'] = STRATEGY_CONFIG['sequence_length_bars']

    try:
        engineer = FeatureEngineer(engineer_init_config)
        logger.info("FeatureEngineer initialized with combined configuration.")
    except Exception as e:
        logger.error(f"An error occurred during FeatureEngineer initialization: {e}", exc_info=True)
        sys.exit(1)


    raw_data_dir = PATHS.get('raw_data_dir')
    raw_data_pattern = PATHS.get('raw_data_pattern')
    processed_data_dir = PATHS.get('processed_data_dir')
    processed_data_pattern = PATHS.get('processed_data_pattern')

    try:
        logger.info(f"Attempting to load raw data for {symbol.upper()} {interval}")
        raw_df = dm.load_data(
            symbol=symbol.upper(),
            interval=interval,
            data_type='raw'
        )
        if raw_df is None or raw_df.empty:
            logger.error("Raw data not found or is empty. Ensure fetch_data.py was run successfully.")
            sys.exit(1)

        logger.info(f"Successfully loaded raw data for {symbol.upper()} {interval}. Shape: {raw_df.shape}")

        logger.info("Starting feature engineering...")
        processed_df = engineer.process(raw_df)

        if processed_df is None or processed_df.empty:
            logger.error("Feature engineering returned an empty DataFrame. Cannot save.")
            sys.exit(1)

        logger.info(f"Saving processed data for {symbol.upper()} {interval}")
        dm.save_data(
            df_to_save=processed_df,
            symbol=symbol.upper(),
            interval=interval,
            data_type='processed',
        )

    except TemporalSafetyError as e:
        logger.error(f"Feature engineering aborted due to temporal safety violation: {str(e)}")
        if hasattr(e, 'features') and e.features:
            logger.error(f"Violating features: {', '.join(e.features)}")
        logger.error("Action required: Inspect the feature engineering logic for the violating features in utils/feature_engineering/feature_engineer.py")
        logger.error("and/or adjust the temporal_validation thresholds in config/feature_config_schema.py if appropriate.")
        sys.exit(1)

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")
        sys.exit(1)
    except ValueError as ve:
        logger.error(f"Value error during feature engineering: {ve}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during feature engineering: {e}", exc_info=True)
        sys.exit(1)


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate features from raw market data and save the result.'
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
        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'], # Match fetch_data choices
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
        logger.error("Feature generation script terminated due to an unhandled error.")
        sys.exit(1)


    """
    Usage example:

    Generate features for BTCUSDT 1-hour data:
        python scripts/generate_features.py --symbol BTCUSDT --interval 1h

    Generate features for ADAUSDT 5-minute data:
        python -m  scripts.generate_features --symbol ADAUSDT --interval 5m

    Ensure you have run the fetch_data script first to obtain the raw data:
        python -m scripts.fetch_data --symbol ADAUSDT --interval 5m --start_dateYYYY-MM-DD

    Ensure config/params.py (with FEATURE_CONFIG and STRATEGY_CONFIG)
    and config/paths.py are correctly configured.
    """
