#!/usr/bin/env python3
"""
generate_features.py

Loads raw OHLCV data using DataManager, engineers features using the FeaturesEngineer,
and saves the resulting DataFrame using DataManager.

Uses the updated configuration structure from config/params.py and config/paths.py.
Configures logging using utils/logger_config.py.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import copy # Import copy for deep copying config
from dotenv import load_dotenv # Import load_dotenv

# --- Load Environment Variables ---
# This should be one of the first things your script does.
load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration and utilities
try:
    # Import configuration parameters
    # Import FEATURE_CONFIG and STRATEGY_CONFIG for merging config
    from config.params import FEATURE_CONFIG, STRATEGY_CONFIG

    # Import centralized path configurations
    from config.paths import PATHS

    # Import the DataManager
    from utils.data_manager import DataManager
    # Import the FeaturesEngineer
    from utils.features_engineer import FeaturesEngineer
    # Assuming TemporalSafetyError is defined in a custom exceptions.py file
    from utils.exceptions import TemporalSafetyError

    # Import setup_rotating_logging
    from utils.logger_config import setup_rotating_logging

except ImportError as e:
    # Use a basic print here as logging might not be fully configured yet if imports fail
    print(f"ERROR: Failed to import necessary modules. Ensure config/, utils/, and exceptions.py are correctly structured. Error: {e}")
    sys.exit(1) # Exit if essential imports fail
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
# Call setup_rotating_logging here at the script entry point
# Use 'generate_features' as the base filename for the log file
setup_rotating_logging('generate_features', logging.INFO)

# Get the logger for this script AFTER setup_rotating_logging has run
# This ensures the logger uses the configured handlers
logger = logging.getLogger(__name__)
logger.info("Rotating logging configured successfully.")


# --- Feature Generation Logic ---
def generate_features_pipeline(symbol: str, interval: str):
    """
    End-to-end feature generation pipeline: loads raw data using DataManager,
    engineers features using the FeaturesEngineer, and saves the resulting
    processed data using DataManager. Handles errors including temporal safety violations.

    Args:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
        interval (str): The time interval for candles (e.g., '1m', '5m', '1h').
    """
    logger.info(f"Starting feature generation pipeline for {symbol.upper()} {interval}...")

    # Instantiate DataManager
    dm = DataManager()

    # --- Create a combined config for FeaturesEngineer ---
    # Merge FEATURE_CONFIG with relevant parameters from STRATEGY_CONFIG
    # Use deepcopy to avoid modifying the original imported configs
    feature_engineer_config = copy.deepcopy(FEATURE_CONFIG)

    # Add/Override parameters required by FeaturesEngineer from STRATEGY_CONFIG
    # Prioritize STRATEGY_CONFIG if available, then FEATURE_CONFIG default
    # Use .get() with default fallbacks to handle cases where keys might be missing
    # NOTE: Ensure these keys are actually used and expected by FeaturesEngineer
    feature_engineer_config['volatility_window_bars'] = STRATEGY_CONFIG.get('volatility_window_bars', feature_engineer_config.get('volatility_window_bars'))
    feature_engineer_config['fixed_take_profit_pct'] = STRATEGY_CONFIG.get('fixed_take_profit_pct', feature_engineer_config.get('fixed_take_profit_pct'))
    feature_engineer_config['fixed_stop_loss_pct'] = STRATEGY_CONFIG.get('fixed_stop_loss_pct', feature_engineer_config.get('fixed_stop_loss_pct'))
    feature_engineer_config['alpha_take_profit'] = STRATEGY_CONFIG.get('alpha_take_profit', feature_engineer_config.get('alpha_take_profit'))
    feature_engineer_config['alpha_stop_loss'] = STRATEGY_CONFIG.get('alpha_stop_loss', feature_engineer_config.get('alpha_stop_loss'))
    feature_engineer_config['trend_filter_enabled'] = STRATEGY_CONFIG.get('trend_filter_enabled', feature_engineer_config.get('trend_filter_enabled'))
    feature_engineer_config['trend_filter_ema_period'] = STRATEGY_CONFIG.get('trend_filter_ema_period', feature_engineer_config.get('trend_filter_ema_period'))
    # NOTE: trend_filter_ema_backward_period is not defined in the provided params.py FEATURE_CONFIG or STRATEGY_CONFIG
    # It might be a leftover or intended for a different strategy. Removing for now or ensure it's in params.py.
    # feature_engineer_config['trend_filter_ema_backward_period'] = STRATEGY_CONFIG.get('trend_filter_ema_backward_period', feature_engineer_config.get('trend_filter_ema_backward_period'))


    # Ensure temporal validation is enabled for feature generation script unless explicitly disabled in STRATEGY_CONFIG
    # Prioritize STRATEGY_CONFIG's temporal_validation setting if it exists
    temp_val_cfg = STRATEGY_CONFIG.get('temporal_validation', FEATURE_CONFIG.get('temporal_validation', {}))
    if 'enabled' not in temp_val_cfg:
         temp_val_cfg['enabled'] = True # Default to enabled if not specified in either config
    feature_engineer_config['temporal_validation'] = temp_val_cfg

    # Add sequence_length_bars from STRATEGY_CONFIG if not in FEATURE_CONFIG (or prioritize STRATEGY_CONFIG)
    # FeaturesEngineer uses this for context, but it's primarily a model/strategy parameter
    # Ensure it's passed if needed by FeaturesEngineer methods (currently not explicitly used in process())
    # If FeaturesEngineer needed it, we'd add it here:
    # feature_engineer_config['sequence_length_bars'] = STRATEGY_CONFIG.get('sequence_length_bars', FEATURE_CONFIG.get('sequence_length_bars'))


    # Instantiate FeaturesEngineer with the combined config
    try:
        engineer = FeaturesEngineer(feature_engineer_config)
        logger.info("FeaturesEngineer initialized with combined configuration.")
    except Exception as e:
        logger.error(f"An error occurred during FeaturesEngineer initialization: {e}", exc_info=True)
        sys.exit(1)


    # Construct paths using the PATHS dictionary (DataManager does this internally now)
    # Use the new lowercase keys from paths.py
    raw_data_dir = PATHS.get('raw_data_dir')
    raw_data_pattern = PATHS.get('raw_data_pattern')
    processed_data_dir = PATHS.get('processed_data_dir')
    processed_data_pattern = PATHS.get('processed_data_pattern')

    # NOTE: The manual path construction below is no longer needed because DataManager.load_data and save_data
    # construct the paths internally using the symbol, interval, and data_type.
    # Keeping the path variables defined above for clarity if needed elsewhere, but removing the manual path string formatting.
    # if not all([raw_data_dir, raw_data_pattern, processed_data_dir, processed_data_pattern]):
    #     logger.error("Missing required paths or patterns in config/paths.py. Ensure 'raw_data_dir', 'raw_data_pattern', 'processed_data_dir', and 'processed_data_pattern' keys are defined.")
    #     sys.exit(1)

    # Ensure raw_data_dir and processed_data_dir are Path objects (DataManager should handle this but good practice)
    # raw_data_dir_path = Path(raw_data_dir)
    # processed_data_dir_path = Path(processed_data_dir)

    # raw_path = raw_data_dir_path / raw_data_pattern.format(...) # Removed manual path construction
    # processed_path = processed_data_dir_path / processed_data_pattern.format(...) # Removed manual path construction


    try:
        # Load raw data using DataManager with the new method signature
        logger.info(f"Attempting to load raw data for {symbol.upper()} {interval}")
        # The new load_data method constructs the path internally
        raw_df = dm.load_data(
            symbol=symbol.upper(),  # Pass the symbol
            interval=interval,      # Pass the interval
            data_type='raw'         # Specify the data type as 'raw'
        )
        # Check if loading was successful and data is not empty
        if raw_df is None or raw_df.empty:
            logger.error("Raw data not found or is empty. Ensure fetch_data.py was run successfully.")
            sys.exit(1) # Exit if no data was loaded

        logger.info(f"Successfully loaded raw data for {symbol.upper()} {interval}. Shape: {raw_df.shape}")


        # Engineer features using the FeaturesEngineer
        logger.info("Starting feature engineering...")
        # The engineer.process method handles NaN cleaning and temporal validation internally
        processed_df = engineer.process(raw_df)

        # Check if feature engineering returned an empty DataFrame
        if processed_df is None or processed_df.empty:
            logger.error("Feature engineering returned an empty DataFrame. Cannot save.")
            sys.exit(1) # Exit if processed_df is empty

        # Save processed data using DataManager with the new method signature
        # The new save_data method constructs the path internally
        logger.info(f"Saving processed data for {symbol.upper()} {interval}")
        dm.save_data(
            df_to_save=processed_df,
            symbol=symbol.upper(),     # Pass the symbol
            interval=interval,         # Pass the interval
            data_type='processed',     # Specify the data type as 'processed'
            # REMOVED: to_parquet=True # This argument is not accepted by DataManager.save_data
        )
        # The DataManager.save_data method logs the success message internally,
        # so this extra log line is redundant.
        # logger.info(f"Successfully saved processed data for {symbol.upper()} {interval}. Shape: {processed_df.shape}")


    except TemporalSafetyError as e:
        # Catch the specific temporal safety error raised by FeaturesEngineer
        logger.error(f"Feature engineering aborted due to temporal safety violation: {str(e)}")
        # Log the specific features that caused the violation if available
        if hasattr(e, 'features') and e.features:
            logger.error(f"Violating features: {', '.join(e.features)}")
        logger.error("Action required: Inspect the feature engineering logic for the violating features in utils/features_engineer.py")
        logger.error("and/or adjust the temporal_validation thresholds in config/params.py if appropriate.")
        sys.exit(1) # Exit the script upon temporal safety violation

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found error: {fnf_error}")
        sys.exit(1)
    except ValueError as ve:
        logger.error(f"Value error during feature engineering: {ve}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors
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
        # Call the main feature generation pipeline function
        generate_features_pipeline(
            symbol=args.symbol,
            interval=args.interval,
        )
    except SystemExit:
        # Catch SystemExit to prevent traceback on intentional sys.exit() calls
        pass
    except Exception:
        # Catch any other exceptions that might have propagated up
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
