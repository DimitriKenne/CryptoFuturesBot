#!/usr/bin/env python3
"""
Script to generate trading labels from processed data (including features)
using the refactored LabelGenerator and labeling strategy pattern.

Supports multiple labeling strategies ('labeling_strategy_1', 'labeling_strategy_2', etc.)
selectable via command-line argument.

Loads processed data (which should include OHLCV and necessary indicators like ATR),
generates labels (1, -1, or 0) based on the chosen labeling strategy and configuration,
saves labeled data (only the 'label' column), and performs basic analysis
of the label distribution, saved to a dedicated folder.

Uses configuration from config/params.py and config/label_config_schema.py.
Configures logging using utils/logger_config.py.
"""

import sys
import argparse
import logging
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
    from config.paths import PATHS
    from config.params import app_config
    from config.label import (
        LabelConfig,
        LabelingStrategy1Config,
        LabelingStrategy2Config,
        LabelingStrategy3Config,
        LabelingStrategy4Config,
    )
    from utils.data_management.data_manager import DataManager
    from utils.labeling.label_generator import LabelGenerator
    from utils.labeling.label_analyzer import LabelAnalyzer
    from utils.logger_config import setup_rotating_logging
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
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to configure rotating logging: {e}. Using basic stdout logging.", exc_info=True)

def create_labels_pipeline(symbol: str, interval: str, labeling_strategy: str):
    """
    Main pipeline to load processed data, generate labels, and save them.

    Args:
        symbol (str): Trading pair symbol (e.g., 'ADAUSDT').
        interval (str): Time interval for candles (e.g., '5m').
        labeling_strategy (str): The name of the labeling strategy to use.
    """
    logger.info(f"Starting labeling pipeline for {symbol} {interval} using '{labeling_strategy}' labeling strategy...")

    # --- 1. Load Configuration from app_config ---
    label_config = copy.deepcopy(app_config.labeling)
    label_config.labeling_strategy_type = labeling_strategy

    logger.info(f"Using labeling configuration: {label_config}")

    # --- 2. Initialize DataManager and LabelGenerator ---
    dm = DataManager()
    try:
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
        logger.info(f"Calculating labels using '{labeling_strategy}' labeling strategy...")
        full_labeled_df_labels_only = gen.calculate_labels(df_input.copy())
        logger.info(f"Successfully generated labels. Labeled DataFrame (labels only) shape: {full_labeled_df_labels_only.shape}")

        # Print calculated TP/SL percentages (if labeling_strategy_1)
        if label_config.labeling_strategy_type == 'labeling_strategy_1':
            strategy_1_config: LabelingStrategy1Config = label_config.labeling_strategy_1
            logger.info(f"\n--- Labeling Strategy 1 (Triple Barrier) Parameters for {symbol} {interval} ---")
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
        fee_param = label_config.trading_fee_rate
        slippage_param = label_config.slippage_tolerance_pct
        future_horizons = getattr(label_config, "analysis_future_horizons", [10, 30, 60, 100, 150])
        strategy_config_obj = getattr(label_config, label_config.labeling_strategy_type)
        f_window_param = getattr(strategy_config_obj, 'future_return_window', 150)

        analyzer = LabelAnalyzer(
            paths=PATHS,
            logger=logger,
            fee=fee_param,
            slippage=slippage_param,
            f_window=f_window_param,
            labeling_strategy_type=label_config.labeling_strategy_type
        )
        analyzer.perform_all_analyses(
            df_combined=df_combined_for_analysis,
            symbol=symbol,
            interval=interval,
            labeling_strategy=labeling_strategy,
            future_horizons=future_horizons
        )
        logger.info("Label analysis complete using LabelAnalyzer.")

    except Exception as e:
        logger.error(f"An error occurred during label analysis: {e}", exc_info=True)
        # Do not sys.exit(1) here, as label generation and saving was successful.

    logger.info(f"Labeling pipeline for {symbol} {interval} completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate trading labels from processed data using various labeling strategies.'
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
        '--labeling-strategy',
        type=str,
        required=True,
        choices=LabelGenerator.get_available_labeling_strategies(),
        help=f'Labeling strategy to use. Available: {", ".join(LabelGenerator.get_available_labeling_strategies())}'
    )

    args = parser.parse_args()

    try:
        create_labels_pipeline(
            symbol=args.symbol,
            interval=args.interval,
            labeling_strategy=args.labeling_strategy,
        )
    except SystemExit:
        pass
    except Exception:
        logger.exception("Labeling script terminated due to an unhandled error.")
        sys.exit(1)

    """
    Usage examples:

    Generate labels using Labeling Strategy 2 (Net Forward Return Quantile):
        python scripts/create_labels.py --symbol BTCUSDT --interval 1h --labeling-strategy labeling_strategy_2

    Generate labels using Labeling Strategy 1 (Triple Barrier):
        python scripts/create_labels.py --symbol ADAUSDT --interval 5m --labeling-strategy labeling_strategy_1

    Generate labels using Labeling Strategy 3 (Future Range Dominance):
        python scripts/create_labels.py --symbol ADAUSDT --interval 15m --labeling-strategy labeling_strategy_3

    Ensure you have processed data files (including necessary features like ATR columns if using triple_barrier)
    in your data/processed directory, and that config/params.py and config/paths.py are correct.
    The feature generation script must produce an ATR column named 'atr_{lookback}'
    (e.g., 'atr_14') matching the 'vol_adj_lookback' parameter in LABELING_CONFIG
    if using Labeling Strategy 1 with volatility adjustment.
    """
