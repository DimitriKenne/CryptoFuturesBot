#!/usr/bin/env python3
"""
Script to generate trading labels from processed data.

Supports multiple labeling strategies selectable via command-line.
Loads processed data, generates labels (1, -1, or 0), saves labeled data,
and performs basic analysis.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import copy
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
    from utils.labeling.analysis_plotter import AnalysisPlotter
    from utils.labeling.analysis_calculator import AnalysisCalculator
    from utils.logger_config import setup_rotating_logging
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import modules. Check project setup. Error: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during imports/config loading: {e}", file=sys.stderr)
    sys.exit(1)

# Set up Logging
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
    Orchestrates strategy-specific analysis during label generation.

    Args:
        symbol (str): Trading pair symbol (e.g., 'ADAUSDT').
        interval (str): Time interval (e.g., '5m').
        labeling_strategy (str): The labeling strategy to use.
    """
    logger.info(f"Starting labeling pipeline for {symbol} {interval} using '{labeling_strategy}' strategy...")

    # Load and configure labeling settings
    label_config = copy.deepcopy(app_config.labeling)
    label_config.labeling_strategy_type = labeling_strategy

    logger.info(f"Using labeling config for strategy: {label_config.labeling_strategy_type}")
    logger.debug(f"Full labeling config: {label_config}")

    # Initialize data manager and analysis tools
    dm = DataManager()
    
    trading_fee_rate = app_config.labeling.trading_fee_pct / 100.0
    slippage_tolerance_rate = app_config.labeling.slippage_tolerance_pct / 100.0
    
    analysis_calculator = AnalysisCalculator(trading_fee_rate, slippage_tolerance_rate)
    
    # Initialize AnalysisPlotter with plot_pattern
    analysis_plotter = AnalysisPlotter(logger, plot_pattern=PATHS.get("labeling_analysis_plot_pattern"))

    try:
        gen = LabelGenerator(config=label_config, logger=logger)
    except Exception as e:
        logger.error(f"Error initializing LabelGenerator: {e}", exc_info=True)
        sys.exit(1)

    # Load processed data
    logger.info(f"Loading processed data (including features) for {symbol} {interval}")
    try:
        df_input = dm.load_data(symbol=symbol, interval=interval, data_type='processed')
        logger.info(f"Loaded processed data. Shape: {df_input.shape}")
    except FileNotFoundError:
        logger.critical(f"Processed data file not found for {symbol} {interval}.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error loading processed data: {e}", exc_info=True)
        sys.exit(1)

    # Validate input DataFrame
    if df_input is None or df_input.empty:
        logger.critical(f"Loaded processed data for {symbol} {interval} is empty. Cannot generate labels.")
        sys.exit(1)
    if not isinstance(df_input.index, pd.DatetimeIndex):
        logger.critical("DataFrame lacks DatetimeIndex. Ensure correct indexing.")
        sys.exit(1)
    if not all(col in df_input.columns for col in ['open', 'high', 'low', 'close']):
        logger.critical("DataFrame missing essential OHLCV columns.")
        sys.exit(1)
    logger.info("Input data validation passed.")

    # Generate Labels and Perform Strategy-Specific Analysis
    try:
        logger.info(f"Calculating labels using '{labeling_strategy}' strategy...")
        
        # Set up strategy-specific analysis output directory
        base_analysis_dir = Path(PATHS.get("analysis_dir", "./results/analysis"))
        strategy_relative_path = PATHS.get("labeling_strategy_analysis_dir_pattern", "").format(labeling_strategy=labeling_strategy)
        strategy_analysis_output_dir = base_analysis_dir / strategy_relative_path
        
        try:
            strategy_analysis_output_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Ensured strategy-specific analysis directory exists: {strategy_analysis_output_dir}")
        except OSError as e:
            logger.error(f"Error creating strategy analysis directory: {e}", exc_info=True)
            raise

        full_labeled_df_labels_only = gen.calculate_labels(
            df=df_input.copy(),
            plotter=analysis_plotter,
            calculator=analysis_calculator,
            output_dir=strategy_analysis_output_dir,
            symbol=symbol,
            interval=interval
            # plot_pattern is now passed via AnalysisPlotter initialization, no longer needed here
        )
        logger.info(f"Labels generated. Labeled DataFrame shape: {full_labeled_df_labels_only.shape}")

        if labeling_strategy == 'labeling_strategy_1':
            strategy_1_config: LabelingStrategy1Config = getattr(label_config, labeling_strategy)
            logger.info(f"\n--- Labeling Strategy 1 Parameters for {symbol} {interval} ---")
            logger.info(f"  Profit Multiplier: {strategy_1_config.profit_multiplier_pct}%")
            logger.info(f"  Stop Loss Multiplier: {strategy_1_config.stop_loss_multiplier_pct}%")
            logger.info(f"  Future Return Window: {strategy_1_config.future_return_window} bars")
            logger.info(f"  Volatility Adjustment Lookback: {strategy_1_config.vol_adj_lookback} bars")
            logger.info(f"----------------------------------------------------\n")

    except ValueError as e:
        logger.critical(f"Error during label generation: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error during labeling: {e}", exc_info=True)
        sys.exit(1)

    # Prepare and save labeled data
    if 'label' not in full_labeled_df_labels_only.columns:
        logger.critical("Label column missing. Cannot save labeled data.")
        sys.exit(1)

    labeled_data_to_save = pd.DataFrame(
        {'label': full_labeled_df_labels_only['label']},
        index=full_labeled_df_labels_only.index
    )
    labeled_data_to_save = labeled_data_to_save.reindex(df_input.index, fill_value=0)
    labeled_data_to_save['label'] = labeled_data_to_save['label'].astype(pd.Int8Dtype())

    logger.info(f"Saving labeled data for {symbol} {interval}...")
    try:
        dm.save_data(
            df_to_save=labeled_data_to_save,
            symbol=symbol,
            interval=interval,
            data_type='labeled',
        )
        logger.info(f"Labeled data saved to {dm.get_file_path(symbol, interval, 'labeled')}")
    except Exception as e:
        logger.critical(f"Error saving labeled data: {e}", exc_info=True)
        sys.exit(1)

    # Prepare combined DataFrame for common analysis
    df_combined_for_analysis = pd.merge(
        df_input,
        labeled_data_to_save,
        left_index=True,
        right_index=True,
        how='inner'
    )

    if df_combined_for_analysis.empty:
        logger.critical("Combined DataFrame for common analysis is empty. Check data alignment.")
        sys.exit(1)
    logger.info(f"Combined processed data with labels for common analysis. Shape: {df_combined_for_analysis.shape}")

    # Perform Common Label Analyses
    logger.info(f"Performing common label analyses for {symbol} {interval} using LabelAnalyzer...")
    try:
        analyzer = LabelAnalyzer(
            paths=PATHS,
            logger=logger,
            fee=None,
            slippage=None,
            f_window=None,
            labeling_strategy_type=labeling_strategy
        )

        analysis_future_horizons = getattr(app_config.labeling, "analysis_future_horizons", [10, 30, 60])
        if not analysis_future_horizons:
            analysis_future_horizons = [10, 30, 60]
            logger.warning("No 'analysis_future_horizons' in config.labeling. Using default [10, 30, 60].")

        analyzer.perform_all_analyses(
            df_combined=df_combined_for_analysis,
            symbol=symbol,
            interval=interval,
            labeling_strategy=labeling_strategy,
            future_horizons=analysis_future_horizons
        )
        logger.info("Common label analyses complete using LabelAnalyzer.")

    except Exception as e:
        logger.error(f"Error during common label analysis: {e}", exc_info=True)

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

    Ensure processed data files (including necessary features like ATR columns if using triple_barrier)
    exist in data/processed, and config/params.py and config/paths.py are correct.
    """
