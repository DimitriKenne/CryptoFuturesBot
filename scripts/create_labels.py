#!/usr/bin/env python3
"""
Script to generate trading labels from processed data (including features)
using the refactored LabelGenerator and strategy pattern.

Supports multiple labeling strategies ('directional_ternary', 'triple_barrier', 'max_return_quantile')
selectable via command-line argument.

Loads processed data (which should include OHLCV and necessary indicators like ATR),
generates labels (1, -1, or 0) based on the chosen strategy and configuration,
saves labeled data (only the 'label' column), and performs basic analysis
of the label distribution, saved to a dedicated folder.

Uses configuration from config/params.py and config/paths.py.
Configures logging using utils/logger_config.py.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import copy # Needed for deep copying config
from dotenv import load_dotenv # Import load_dotenv
from typing import Dict, Any, Optional # Import Dict, Any, Optional

# --- Load Environment Variables ---
# This should be one of the first things your script does.
# It looks for a .env file in the current directory or parent directories
# and loads the key-value pairs into the environment.
load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration and utilities
try:
    from config.paths import PATHS
    from config.params import LABELING_CONFIG, STRATEGY_CONFIG # Import STRATEGY_CONFIG for volatility_regime_filter_enabled

    from utils.data_manager import DataManager
    from utils.label_generator import LabelGenerator
    from utils.label_analyzer import LabelAnalyzer # NEW: Import LabelAnalyzer
    from utils.logger_config import setup_rotating_logging
    # from utils.exceptions import TemporalSafetyError # Not directly caught here, handled by generic Exception

except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import necessary modules. "
          f"Ensure your project structure and dependencies are correct. Error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during initial imports or configuration loading: {e}", file=sys.stderr)
    sys.exit(1)


# --- Set up Logging ---
try:
    setup_rotating_logging('create_labels')
    logger = logging.getLogger(__name__)
    logger.info("Rotating logging configured successfully.")
except Exception as e:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(lineno)d]',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__) # Get logger even if basic config used
    logger.warning(f"Failed to configure rotating logging: {e}. Using basic stdout logging.", exc_info=True)


def create_labels_pipeline(symbol: str, interval: str, label_strategy: str):
    """
    Main pipeline to load processed data, generate labels, and save them.

    Args:
        symbol (str): Trading pair symbol (e.g., 'ADAUSDT').
        interval (str): Time interval for candles (e.g., '5m').
        label_strategy (str): The name of the labeling strategy to use.
    """
    logger.info(f"Starting labeling pipeline for {symbol} {interval} using '{label_strategy}' strategy...")

    # --- 1. Load Configuration ---
    # Create a deep copy of LABELING_CONFIG to avoid modifying the original
    labeling_config = copy.deepcopy(LABELING_CONFIG)
    labeling_config['label_type'] = label_strategy

    # Merge volatility regime filter parameters if enabled
    if STRATEGY_CONFIG.get('volatility_regime_filter_enabled', False):
        labeling_config['volatility_regime_max_holding_bars'] = STRATEGY_CONFIG.get('volatility_regime_max_holding_bars')
        labeling_config['allow_trading_in_volatility_regime'] = STRATEGY_CONFIG.get('allow_trading_in_volatility_regime')
        logger.info("Volatility regime filter parameters merged into labeling config.")

    logger.info(f"Using labeling configuration: {labeling_config}")

    # --- 2. Initialize DataManager and LabelGenerator ---
    dm = DataManager()
    try:
        # Pass the logger instance to LabelGenerator
        gen = LabelGenerator(config=labeling_config, logger=logger)
    except Exception as e:
        logger.error(f"An unexpected error occurred initializing LabelGenerator: {e}")
        sys.exit(1)


    # --- 3. Load Processed Data ---
    logger.info(f"Attempting to load processed data (including features) for {symbol} {interval}")
    try:
        df_input = dm.load_data(
            symbol=symbol,
            interval=interval,
            data_type='processed'
        )
        logger.info(f"Successfully loaded processed data for {symbol} {interval}. Shape: {df_input.shape}")
    except FileNotFoundError:
        logger.critical(f"Processed data file not found for {symbol} {interval}. "
                        f"Please run 'python -m scripts.generate_features --symbol {symbol} --interval {interval}' first.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error loading processed data for {symbol} {interval}: {e}", exc_info=True)
        sys.exit(1)

    # Basic validation of loaded DataFrame
    if df_input.empty:
        logger.critical(f"Loaded processed data for {symbol} {interval} is empty. Cannot generate labels.")
        sys.exit(1)
    if not isinstance(df_input.index, pd.DatetimeIndex):
        logger.critical("Loaded DataFrame does not have a DatetimeIndex. Please ensure your data processing pipeline sets the index correctly.")
        sys.exit(1)
    if not all(col in df_input.columns for col in ['open', 'high', 'low', 'close']):
        logger.critical("Loaded DataFrame is missing essential OHLCV columns (open, high, low, close).")
        sys.exit(1)
    logger.info("Input data basic validation passed.")


    # --- 4. Generate Labels ---
    try:
        logger.info(f"Calculating labels using '{label_strategy}' strategy...")
        # Pass a copy to avoid unintended modifications within the LabelGenerator
        # The returned full_labeled_df should have the correct index from df_input
        # This full_labeled_df will only contain the 'label' column and the index.
        full_labeled_df_labels_only = gen.calculate_labels(df_input.copy())
        logger.info(f"Successfully generated labels. Labeled DataFrame (labels only) shape: {full_labeled_df_labels_only.shape}")
    except ValueError as e:
        logger.critical(f"Error during label generation: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during labeling: {e}", exc_info=True)
        sys.exit(1)

    # --- 5. Prepare Combined DataFrame for Analysis and Saving ---
    # CRITICAL FIX: Merge df_input (with all OHLCV and features) with the generated labels
    # to create a comprehensive DataFrame for analysis.
    if 'label' not in full_labeled_df_labels_only.columns:
        logger.critical("Label column not found in generated labels. Cannot proceed.")
        sys.exit(1)

    # Ensure the label column has the correct index and name
    labeled_data_to_save = pd.DataFrame(
        {'label': full_labeled_df_labels_only['label']},
        index=df_input.index # Use the index from the original processed data
    )
    # Reindex one more time to ensure perfect alignment and fill NaNs if any labels were dropped
    labeled_data_to_save = labeled_data_to_save.reindex(df_input.index, fill_value=0)

    # Merge df_input (all original data and features) with the new 'label' column
    # This df_combined will be used for analysis
    df_combined_for_analysis = pd.merge(
        df_input, # Contains all OHLCV and features
        labeled_data_to_save, # Contains only the 'label' column with the correct index
        left_index=True,
        right_index=True,
        how='inner' # Ensure only matching indices are kept
    )

    if df_combined_for_analysis.empty:
        logger.critical("Combined DataFrame for analysis is empty after merging labels. Check data alignment.")
        sys.exit(1)
    logger.info(f"Successfully combined processed data with labels for analysis. Combined shape: {df_combined_for_analysis.shape}")


    # --- 6. Save Labeled Data ---
    logger.info(f"Attempting to save labeled data for {symbol} {interval}...")
    try:
        # User requested to save labeled data WITHOUT strategy-specific suffix
        dm.save_data(
            df_to_save=labeled_data_to_save, # Save only the 'label' column
            symbol=symbol,
            interval=interval,
            data_type='labeled',
            # name_suffix=f'_{label_strategy}' # REMOVED: User requested no strategy suffix for labeled file
        )
        # The filename will now be like ADAUSDT_5m_labeled.parquet
        logger.info(f"Successfully saved labeled data to {dm.get_file_path(symbol, interval, 'labeled')}") # Updated log message
    except Exception as e:
        logger.critical(f"Error saving labeled data: {e}", exc_info=True)
        sys.exit(1)

    # --- 7. Perform Label Analysis and Plotting using LabelAnalyzer ---
    logger.info(f"Performing label analysis for {symbol} {interval} using LabelAnalyzer...")
    try:
        # Instantiate LabelAnalyzer
        analyzer = LabelAnalyzer(paths=PATHS, logger=logger)
        # Pass the df_combined_for_analysis (which contains OHLCV, features, and the 'label' column)
        # to the analyzer for comprehensive analysis.
        analyzer.perform_all_analyses(
            df_combined=df_combined_for_analysis, # Pass the comprehensive DataFrame
            symbol=symbol,
            interval=interval,
            label_strategy=label_strategy, # Pass the strategy name for folder creation (analysis output still uses it)
            future_horizons=STRATEGY_CONFIG.get('analysis_future_horizons', [5, 10, 20, 50, 100]) # Use config or default
        )
        logger.info("Label analysis complete using LabelAnalyzer.")

    except Exception as e:
        logger.error(f"An error occurred during label analysis: {e}", exc_info=True)
        # Do not sys.exit(1) here, as label generation and saving was successful.
        # Analysis is a secondary step.

    logger.info(f"Labeling pipeline for {symbol} {interval} completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate trading labels from processed data using various strategies.'
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
    parser.add_argument(
        '--label-strategy',
        type=str,
        required=True,
        choices=LabelGenerator.get_available_strategies(), # Dynamically get choices from LabelGenerator
        help=f'Labeling strategy to use. Available: {", ".join(LabelGenerator.get_available_strategies())}'
    )

    args = parser.parse_args()

    try:
        create_labels_pipeline(
            symbol=args.symbol,
            interval=args.interval,
            label_strategy=args.label_strategy,
        )
    except SystemExit:
         pass # Prevent traceback on intentional sys.exit()
    except Exception:
        logger.exception("Labeling script terminated due to an unhandled error.") # Log full traceback
        sys.exit(1)

    """
    Usage examples:

    Generate labels using the simple directional strategy:
        python scripts/create_labels.py --symbol BTCUSDT --interval 1h --label-strategy directional_ternary

    Generate labels using the triple barrier strategy:
        python scripts/create_labels.py --symbol ADAUSDT --interval 5m --label-strategy triple_barrier

    Generate labels using the max return quantile strategy:
        python scripts/create_labels.py --symbol ADAUSDT --interval 15m --label-strategy max_return_quantile

    Ensure you have processed data files (including necessary features like ATR columns if using triple_barrier)
    in your data/processed directory, and that config/params.py and config/paths.py are correct.
    The feature generation script must produce an ATR column named 'atr_{lookback}'
    (e.g., 'atr_14') matching the 'vol_adj_lookback' parameter in LABELING_CONFIG
    if using the 'triple_barrier' strategy with volatility adjustment.
    """
