#!/usr/bin/env python3
"""
Script to generate trading labels from processed data (including features)
using the refactored LabelGenerator and strategy pattern.

Supports multiple labeling strategies ('strategy_1', 'strategy_2', etc.)
selectable via command-line argument.

Loads processed data (which should include OHLCV and necessary indicators like ATR),
generates labels (1, -1, or 0) based on the chosen strategy and configuration,
saves labeled data (only the 'label' column), and performs basic analysis
of the label distribution, saved to a dedicated folder.

Uses configuration from config/label_config_schema.py, config/params.py and config/paths.py.
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
    from utils.labeling.label_generator import LabelGenerator # Updated import path
    from utils.labeling.label_analyzer import LabelAnalyzer # Updated import path
    from utils.logger_config import setup_rotating_logging
    # No need to import specific strategy for type checking here, as LabelConfig is the source of truth

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
    # Start with a deep copy of the default LabelConfig dataclass instance
    label_config = copy.deepcopy(DEFAULT_LABEL_CONFIG)
    # Start with a deep copy of the default StrategyConfig dataclass instance (to get overrides)
    strategy_config = copy.deepcopy(DEFAULT_STRATEGY_CONFIG)

    # Apply command-line override for label_type
    label_config.label_type = label_strategy

    # Apply common parameters from StrategyConfig to label_config
    label_config.trading_fee_rate = strategy_config.trading_fee_rate
    label_config.slippage_tolerance_pct = strategy_config.slippage_tolerance_pct

    logger.info(f"Using labeling configuration: {label_config}")
    logger.info(f"Using strategy configuration (for common parameters): {strategy_config}")

    # --- 2. Initialize DataManager and LabelGenerator ---
    dm = DataManager()
    try:
        # Pass the configured LabelConfig instance directly to LabelGenerator
        gen = LabelGenerator(config=label_config, logger=logger)
    except Exception as e:
        logger.error(f"An unexpected error occurred initializing LabelGenerator: {e}", exc_info=True)
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
        full_labeled_df_labels_only = gen.calculate_labels(df_input.copy())
        logger.info(f"Successfully generated labels. Labeled DataFrame (labels only) shape: {full_labeled_df_labels_only.shape}")
        
        # --- Print Calculated TP/SL Percentages (if Strategy 1) ---
        # Access the strategy-specific config directly from the LabelGenerator's internal config
        if label_config.label_type == 'strategy_1': # Check against strategy type
            strategy_1_config: Strategy1Config = label_config.strategy_1
            logger.info(f"\n--- Strategy 1 (Triple Barrier) Parameters for {symbol} {interval} ---")
            logger.info(f"  Profit Multiplier: {strategy_1_config.profit_multiplier}")
            logger.info(f"  Stop Loss Multiplier: {strategy_1_config.stop_loss_multiplier}")
            logger.info(f"  Future Return Window: {strategy_1_config.future_return_window} bars")
            logger.info(f"  Volatility Adjustment Lookback (ATR): {strategy_1_config.vol_adj_lookback} bars")
            logger.info(f"----------------------------------------------------\n")

    except ValueError as e:
        logger.critical(f"Error during label generation: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during labeling: {e}", exc_info=True)
        sys.exit(1)

    # --- 5. Prepare Combined DataFrame for Analysis and Saving ---
    if 'label' not in full_labeled_df_labels_only.columns:
        logger.critical("Label column not found in generated labels. Cannot proceed.")
        sys.exit(1)

    labeled_data_to_save = pd.DataFrame(
        {'label': full_labeled_df_labels_only['label']},
        index=df_input.index
    )
    # Ensure all original rows are present, filling NaNs (e.g., from initial lookback) with 0
    labeled_data_to_save = labeled_data_to_save.reindex(df_input.index, fill_value=0)

    df_combined_for_analysis = pd.merge(
        df_input,
        labeled_data_to_save,
        left_index=True,
        right_index=True,
        how='inner'
    )

    if df_combined_for_analysis.empty:
        logger.critical("Combined DataFrame for analysis is empty after merging labels. Check data alignment.")
        sys.exit(1)
    logger.info(f"Successfully combined processed data with labels for analysis. Combined shape: {df_combined_for_analysis.shape}")


    # --- 6. Save Labeled Data ---
    logger.info(f"Attempting to save labeled data for {symbol} {interval}...")
    try:
        dm.save_data(
            df_to_save=labeled_data_to_save,
            symbol=symbol,
            interval=interval,
            data_type='labeled',
        )
        logger.info(f"Successfully saved labeled data to {dm.get_file_path(symbol, interval, 'labeled')}")
    except Exception as e:
        logger.critical(f"Error saving labeled data: {e}", exc_info=True)
        sys.exit(1)

    # --- 7. Perform Label Analysis and Plotting using LabelAnalyzer ---
    logger.info(f"Performing label analysis for {symbol} {interval} using LabelAnalyzer...")
    try:
        # Use values directly from the final_label_config instance for LabelAnalyzer
        fee_param = label_config.trading_fee_rate
        slippage_param = label_config.slippage_tolerance_pct
        
        # Get f_window from the specific strategy config within label_config
        strategy_config_obj = getattr(label_config, label_config.label_type)
        # Default to a safe value if 'future_return_window' is not found (shouldn't happen with proper schema)
        f_window_param = getattr(strategy_config_obj, 'future_return_window', 150) # Use a reasonable default or fetch from general config if needed

        analyzer = LabelAnalyzer(paths=PATHS, logger=logger, fee=fee_param, slippage=slippage_param, f_window=f_window_param)
        analyzer.perform_all_analyses(
            df_combined=df_combined_for_analysis,
            symbol=symbol,
            interval=interval,
            label_strategy=label_strategy,
            future_horizons=strategy_config.analysis_future_horizons
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

    Generate labels using Strategy 2 (Net Forward Return Quantile):
        python scripts/create_labels.py --symbol BTCUSDT --interval 1h --label-strategy strategy_2

    Generate labels using Strategy 1 (Triple Barrier):
        python scripts/create_labels.py --symbol ADAUSDT --interval 5m --label-strategy strategy_1

    Generate labels using Strategy 3 (Future Range Dominance):
        python scripts/create_labels.py --symbol ADAUSDT --interval 15m --label-strategy strategy_3

    Ensure you have processed data files (including necessary features like ATR columns if using triple_barrier)
    in your data/processed directory, and that config/params.py and config/paths.py are correct.
    The feature generation script must produce an ATR column named 'atr_{lookback}'
    (e.g., 'atr_14') matching the 'vol_adj_lookback' parameter in LABELING_CONFIG
    if using Strategy 1 with volatility adjustment.
    """
