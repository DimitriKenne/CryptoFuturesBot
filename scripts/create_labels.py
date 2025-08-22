#!/usr/bin/env python3
"""
Script to generate trading labels from processed data.

This script acts as the main entry point for the labeling pipeline. It validates
the application configuration using the central validator, initializes all necessary
components, orchestrates the label generation process, and triggers all associated analyses.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import configuration and refactored components ---
try:
    from config.params import app_config
    from config.validator import validate_config  # CORRECT: Import the central validator
    from utils.data_management.data_manager import DataManager
    from utils.labeling.label_generator import LabelGenerator, LABELING_STRATEGY_MAP
    from utils.labeling.label_analyzer import LabelAnalyzer
    from utils.labeling.analysis_plotter import AnalysisPlotter
    from utils.labeling.analysis_calculator import AnalysisCalculator
    from utils.logger_config import setup_rotating_logging
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import modules. Check project setup. Error: {e}", file=sys.stderr)
    sys.exit(1)

# Set up Logging
setup_rotating_logging('create_labels')
logger = logging.getLogger(__name__)

def create_labels_pipeline(symbol: str, interval: str, labeling_strategy_name: str):
    """
    Main pipeline to load data, generate labels, and save results and analyses.
    """
    logger.info(f"--- Starting Labeling Pipeline for {symbol} {interval} using '{labeling_strategy_name}' ---")

    # --- 0. CONFIGURATION VALIDATION ---
    logger.info("Validating labeling configuration...")
    try:
        # CORRECT: Use the central validate_config function
        validate_config(app_config.labeling)
        logger.info("Configuration validated successfully.")
    except (ValueError, TypeError) as e:
        logger.critical(f"Labeling configuration is invalid: {e}", exc_info=True)
        sys.exit(1)

    # --- 1. INITIALIZATION PHASE (WIRING THE COMPONENTS) ---
    logger.info("Initializing components...")
    try:
        # Core Utilities (created once)
        dm = DataManager()
        plotter = AnalysisPlotter(logger=logger)
        
        # Use the validated app_config.labeling object
        fee_rate = app_config.labeling.trading_fee_pct / 100.0
        slippage_rate = app_config.labeling.slippage_tolerance_pct / 100.0
        calculator = AnalysisCalculator(trading_fee_rate=fee_rate, slippage_tolerance_rate=slippage_rate)

        # Dynamically select and instantiate the strategy
        if labeling_strategy_name not in LABELING_STRATEGY_MAP:
            raise ValueError(f"Strategy '{labeling_strategy_name}' not found.")
        
        StrategyClass = LABELING_STRATEGY_MAP[labeling_strategy_name]
        strategy_config = getattr(app_config.labeling, labeling_strategy_name)
        
        labeling_strategy = StrategyClass(
            config=strategy_config,
            logger=logger,
            trading_fee_rate=fee_rate,
            slippage_tolerance_rate=slippage_rate
        )

        # Instantiate the main orchestrators with their dependencies
        label_generator = LabelGenerator(
            labeling_strategy=labeling_strategy,
            min_holding_period=app_config.labeling.min_holding_period,
            logger=logger
        )
        
        label_analyzer = LabelAnalyzer(
            data_manager=dm,
            calculator=calculator,
            plotter=plotter
        )
        logger.info("All components initialized successfully.")

    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. EXECUTION PHASE (ORCHESTRATING THE WORKFLOW) ---
    # Load Data
    logger.info(f"Loading processed data for {symbol} {interval}...")
    df_input = dm.load_data(symbol=symbol, interval=interval, data_type='processed')
    if df_input is None or df_input.empty:
        logger.critical("Processed data is empty or not found. Exiting.")
        sys.exit(1)
    logger.info(f"Loaded processed data with shape: {df_input.shape}")

    # Generate Labels & Strategy-Specific Analysis
    logger.info("Generating labels and running strategy-specific analysis...")
    df_labeled = label_generator.calculate_labels(
        df=df_input.copy(),
        dm=dm,
        plotter=plotter,
        calculator=calculator,
        symbol=symbol,
        interval=interval
    )
    logger.info("Label generation and strategy-specific analysis complete.")

    # Save the final labeled data (labels only) for compatibility
    if 'label' not in df_labeled.columns:
        logger.critical("Label generation failed to produce a 'label' column. Exiting.")
        sys.exit(1)
        
    labeled_data_to_save = pd.DataFrame(df_labeled['label'])
    dm.save_data(
        df_to_save=labeled_data_to_save,
        symbol=symbol,
        interval=interval,
        data_type='labeled'
    )
    logger.info(f"Final labeled data saved successfully.")

    # Perform and Save Common Analyses
    logger.info("Performing common analyses on the newly generated labels...")
    df_combined_for_analysis = df_input.join(labeled_data_to_save, how='inner')
    
    f_window = getattr(strategy_config, "future_return_window", 150)
    analysis_horizons = app_config.labeling.analysis_future_horizons or [10, 30, 60]

    label_analyzer.perform_all_analyses(
        df_combined=df_combined_for_analysis,
        symbol=symbol,
        interval=interval,
        labeling_strategy=labeling_strategy.__class__.__name__,
        future_horizons=analysis_horizons,
        f_window=f_window
    )
    logger.info("Common analyses complete. All artifacts saved.")
    logger.info(f"--- Labeling Pipeline for {symbol} {interval} finished successfully! ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate trading labels from processed data.')
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, required=True, help='Time interval for candles (e.g., 5m, 1h)')
    parser.add_argument('--labeling-strategy', type=str, required=True, choices=LABELING_STRATEGY_MAP.keys(), help='Labeling strategy to use.')
    args = parser.parse_args()

    try:
        create_labels_pipeline(
            symbol=args.symbol,
            interval=args.interval,
            labeling_strategy_name=args.labeling_strategy,
        )
    except SystemExit:
        pass
    except Exception as e:
        logger.exception("Labeling script terminated due to an unhandled error.")
        sys.exit(1)