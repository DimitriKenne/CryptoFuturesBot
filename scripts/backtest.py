#!/usr/bin/env python3
"""
backtest.py

Orchestrates the backtesting process for a trained trading model.

Steps:
1. Parses command-line arguments.
2. Sets up logging.
3. Loads processed data using DataManager.
4. Splits data based on backtest mode.
5. Loads the trained model using ModelTrainer (which uses DataManager).
6. Cleans backtest_data to remove rows with NaNs in relevant features.
7. Generates predictions AND probability scores using the loaded model on cleaned data.
8. Initializes and runs the BacktestSimulator simulation, passing predictions and probabilities.
9. Runs the ResultAnalyser.
10. Saves backtest results using the BacktestSimulator's internal saving mechanism.

MODIFIED (NEW): Now uses the comprehensive `app_config` object from `config.params`.
MODIFIED: Updated argument parsing to retrieve default values from `app_config`.
MODIFIED: Passed `app_config.strategy`, `app_config.backtest`, `app_config.exchange`,
          and `app_config.features` dataclass instances directly to the `BacktestSimulator`
          constructor for improved configuration management.
FIXED: Ensured predictions and probabilities are correctly aligned and handled for empty data.
UPDATED: Renamed Backtester to BacktestSimulator to reflect the new class name.
UPDATED: Changed ResultsAnalyser to ResultAnalyser.
UPDATED: Corrected model choices to use AVAILABLE_MODEL_TYPES from ModelConfig.
FIXED: Removed incorrect `get_model_config` call; `app_config.model` is directly passed to ModelTrainer.
FIXED: Corrected keyword argument for ModelTrainer initialization from 'config' to 'model_config'.
FIXED: Corrected the preprocessor check to use 'trainer.preprocessor_builder.preprocessor'.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import copy # Import copy for deep copying dataframes/series

# --- Determine Project Root ---
try:
    script_dir = Path(__file__).resolve().parent
    PROJECT_ROOT = script_dir.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()
    print(f"Warning: __file__ not defined. Assuming project root is current directory: {PROJECT_ROOT}", file=sys.stderr)

# Add project root to sys.path BEFORE attempting project imports
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import Project Modules ---
try:
    # Import the aggregated application configuration
    from config.params import app_config

    # Import centralized path configurations
    from config.paths import PATHS

    # Import the DataManager
    from utils.data_manager import DataManager
    # Import the ModelTrainer
    from utils.training.model_trainer import ModelTrainer

    # Import the BacktestSimulator (renamed from Backtester)
    from utils.backtest_engine.simulator import BacktestSimulator # Updated import path

    # Import the logging setup function
    from utils.logger_config import setup_rotating_logging

    # Import the ResultAnalyser (updated class name)
    try:
        from utils.result_analyzer import ResultAnalyser # Updated import and class name
        RESULTS_ANALYSER_AVAILABLE = True
    except ImportError:
        RESULTS_ANALYSER_AVAILABLE = False
        print("WARNING: ResultAnalyser class not found in utils. Skipping detailed analysis.", file=sys.stderr)


except ImportError as e:
    print(f"ERROR: Failed to import necessary project modules: {e}", file=sys.stderr)
    print("Please ensure config/, utils/ directories exist within the project root and contain the required files.", file=sys.stderr)
    sys.exit(1)
except FileNotFoundError as e:
    print(f"ERROR: Configuration file not found: {e}. Ensure config/params.py and config/paths.py exist.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during initial imports: {e}", exc_info=True, file=sys.stderr) # Added exc_info for more details
    sys.exit(1)


# --- Global Logger Setup ---
setup_rotating_logging("backtest")
logger = logging.getLogger(__name__)


# --- Main Backtest Pipeline Function ---

def run_backtest_pipeline(symbol: str, interval: str, model_key: str, backtest_mode: str = 'test', train_ratio: float = 0.8):
    """
    Runs the complete backtesting pipeline.

    Args:
        symbol (str): Trading pair symbol.
        interval (str): Data interval.
        model_key (str): Key for the model configuration.
        backtest_mode (str): 'full', 'train', or 'test'.
        train_ratio (float): Train/test split ratio.
    """
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n--- Starting Backtest Run ({run_timestamp}) ---")
    logger.info(f"Symbol: {symbol.upper()}, Interval: {interval}, Model: {model_key}")
    logger.info(f"Backtest Mode: {backtest_mode}, Train Ratio: {train_ratio}")
    logger.info("-" * 40)

    dm = DataManager()

    # --- Load Processed Data using DataManager ---
    logger.info("Loading processed data using DataManager...")
    try:
        data = dm.load_data(symbol=symbol, interval=interval, data_type='processed')

        if data is None or data.empty:
            raise FileNotFoundError(f"Processed data is empty or could not be loaded for {symbol} {interval}.")

        if not isinstance(data.index, pd.DatetimeIndex):
             logger.warning("Data index is not DatetimeIndex. Attempting conversion.")
             data.index = pd.to_datetime(data.index, utc=True)
        elif data.index.tz is None:
             logger.warning("Data index is timezone naive. Assuming UTC.")
             data.index = data.index.tz_localize('UTC')
        elif str(data.index.tz) != 'UTC':
             logger.warning(f"Data index has timezone {data.index.tz}. Converting to UTC.")
             data.index = data.index.tz_convert('UTC')

        if not data.index.is_monotonic_increasing:
             logger.warning("Data index is not monotonic increasing. Sorting...")
             data.sort_index(inplace=True)

        logger.info(f"Data loaded successfully: {len(data)} rows, Index: {data.index.min()} to {data.index.max()}")
    except FileNotFoundError as fnf:
        logger.critical(f"Processed data file not found or empty: {fnf}. Run data processing first.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error loading or processing data: {e}", exc_info=True)
        sys.exit(1)

    # --- Split Data for Backtesting ---
    logger.info(f"Splitting data for backtest mode: '{backtest_mode}'...")
    original_backtest_data_len = 0
    backtest_data = pd.DataFrame() # Initialize empty DataFrame
    try:
        if backtest_mode == 'full':
            backtest_data = data.copy()
        elif backtest_mode in ['train', 'test']:
            if not (0 < train_ratio < 1):
                 raise ValueError(f"Invalid train_ratio: {train_ratio}. Must be between 0 and 1 (exclusive).")

            train_size = int(len(data) * train_ratio)
            # Ensure at least one bar for train and test if possible
            if len(data) < 2:
                raise ValueError(f"Insufficient data points ({len(data)}) for train/test split. Need at least 2.")
            
            if train_size == 0:
                train_size = 1 # Ensure at least one train bar
            if train_size >= len(data):
                train_size = len(data) - 1 # Ensure at least one test bar

            if backtest_mode == 'train':
                backtest_data = data.iloc[:train_size].copy()
            else: # test mode
                backtest_data = data.iloc[train_size:].copy()
        else:
            raise ValueError(f"Invalid backtest_mode: {backtest_mode}. Must be 'full', 'train', or 'test'.")

        if backtest_data.empty:
            raise ValueError(f"Backtest data split resulted in an empty DataFrame for mode '{backtest_mode}'. Check data length and train_ratio.")

        original_backtest_data_len = len(backtest_data)
        logger.info(f"Using {len(backtest_data)} rows for backtesting from {backtest_data.index.min()} to {backtest_data.index.max()} (before feature cleaning).")
    except Exception as e:
        logger.critical(f"Error splitting data: {e}", exc_info=True)
        sys.exit(1)

    # --- Load Trained Model using ModelTrainer ---
    logger.info(f"Loading trained model '{model_key}' using ModelTrainer...")
    trainer: ModelTrainer # Declare type for trainer
    try:
        # FIX: Directly pass app_config.model, using 'model_config' keyword argument
        trainer = ModelTrainer(model_config=app_config.model) 
        trainer.load(symbol=symbol, interval=interval, model_key=model_key)

        if not hasattr(trainer, 'model') and not hasattr(trainer, 'pipeline'):
             raise RuntimeError("ModelTrainer failed to load model or pipeline.")
        # FIX: Corrected preprocessor check to use trainer.preprocessor_builder.preprocessor
        if not hasattr(trainer, 'preprocessor_builder') or trainer.preprocessor_builder.preprocessor is None:
            raise RuntimeError("ModelTrainer failed to load the preprocessor.")
        if not hasattr(trainer, 'feature_columns_original') or not trainer.feature_columns_original:
            logger.warning("Original feature columns not found in loaded model metadata. This might affect data cleaning for predictions.")
            trainer.feature_columns_original = app_config.model.features_to_use # Fallback using explicit config
            if not trainer.feature_columns_original:
                 raise RuntimeError("Could not determine original feature columns used by the model from metadata or config.")
            logger.info(f"Using feature_columns_original from config as fallback: {trainer.feature_columns_original}")


        logger.info(f"Model '{model_key}' loaded successfully.")
    except FileNotFoundError:
        logger.critical(f"Trained model file not found for {symbol} {interval} {model_key}. Train the model first using train_model.py.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Error loading trained model using ModelTrainer: {e}", exc_info=True)
        sys.exit(1)

    # --- Clean backtest_data Features Before Prediction ---
    logger.info("Cleaning backtest_data features before prediction...")
    model_predictions = pd.Series(dtype=np.int8, index=pd.DatetimeIndex([]))
    model_probabilities = pd.DataFrame(dtype=float, index=pd.DatetimeIndex([]))
    try:
        model_feature_cols = trainer.feature_columns_original
        if not model_feature_cols:
            raise ValueError("Original feature columns used by the model are not available from the loaded trainer. Cannot clean data for prediction.")

        missing_cols_in_backtest_data = [col for col in model_feature_cols if col not in backtest_data.columns]
        if missing_cols_in_backtest_data:
            logger.warning(f"Backtest data is missing expected feature columns: {missing_cols_in_backtest_data}. These will be ignored during cleaning.")
            model_feature_cols = [col for col in model_feature_cols if col in backtest_data.columns] # Filter to only present columns
            if not model_feature_cols:
                 raise ValueError("No usable feature columns found in backtest_data that match model's expected features after filtering.")

        # Ensure that `backtest_data` is only dropped on relevant columns.
        # Make a copy to avoid SettingWithCopyWarning if `backtest_data` was a slice.
        backtest_data_for_prediction = backtest_data.copy()
        initial_prediction_rows = len(backtest_data_for_prediction)
        
        # Drop rows with NaNs in any of the model's *actual* feature columns present in the data.
        backtest_data_for_prediction.dropna(subset=model_feature_cols, inplace=True)

        rows_removed_cleaning = initial_prediction_rows - len(backtest_data_for_prediction)
        if rows_removed_cleaning > 0:
            logger.info(f"Removed {rows_removed_cleaning} rows from backtest_data due to NaNs in model feature columns: {model_feature_cols}")

        if backtest_data_for_prediction.empty:
            logger.warning("Backtest data is empty after cleaning NaNs in feature columns. No predictions will be made.")
        else:
            logger.info(f"Backtest data cleaned. Shape after cleaning: {backtest_data_for_prediction.shape}")
            # Use the cleaned data for predictions
            # --- Generate Predictions and Probabilities ---
            logger.info("Generating predictions and probability scores using the loaded model on cleaned data...")
            model_predictions = trainer.predict(backtest_data_for_prediction)
            model_probabilities = trainer.predict_proba(backtest_data_for_prediction)

            if not isinstance(model_predictions, pd.Series):
                logger.warning(f"Model prediction output is not a pandas Series (type: {type(model_predictions)}). Attempting conversion.")
                model_predictions = pd.Series(model_predictions, index=backtest_data_for_prediction.index)

            if not model_predictions.index.equals(backtest_data_for_prediction.index):
                 logger.warning("Prediction index does not match cleaned backtest data index. Reindexing and filling missing with 0.")
                 model_predictions = model_predictions.reindex(backtest_data_for_prediction.index).fillna(0).astype(int)
            
            # Fill potential NaNs in predictions (can happen with LSTM for initial bars or reindexing)
            if model_predictions.isnull().any():
                 logger.warning("NaN values found in predictions after alignment. Filling with 0 (neutral).")
                 model_predictions.fillna(0, inplace=True)

            model_predictions = model_predictions.clip(-1, 1).astype(int)

            if model_probabilities is None or model_probabilities.empty:
                 logger.warning("Model predict_proba returned None or empty DataFrame. Confidence scores will not be used.")
                 model_probabilities = pd.DataFrame(dtype=float, index=backtest_data_for_prediction.index)
            else:
                 if not isinstance(model_probabilities, pd.DataFrame):
                      logger.warning(f"Model probability output is not a pandas DataFrame (type: {type(model_probabilities)}). Attempting conversion.")
                      model_probabilities = pd.DataFrame(model_probabilities, index=backtest_data_for_prediction.index)

                 if not model_probabilities.index.equals(backtest_data_for_prediction.index):
                      logger.warning("Probability index does not match cleaned backtest data index. Reindexing and filling missing with NaN.")
                      model_probabilities = model_probabilities.reindex(backtest_data_for_prediction.index)

                 expected_proba_cols = [-1, 0, 1]
                 if not all(col in model_probabilities.columns for col in expected_proba_cols):
                      logger.warning(f"Probability DataFrame is missing expected columns ({expected_proba_cols}). Found: {model_probabilities.columns.tolist()}. Confidence scores may be unreliable.")
                      for col in expected_proba_cols:
                           if col not in model_probabilities.columns:
                                model_probabilities[col] = np.nan

                 if model_probabilities.isnull().any().any():
                      logger.warning("NaN values found in probabilities after alignment. Filling with NaN.")


            logger.info(f"Predictions generated. Shape: {model_predictions.shape}")
            logger.debug(f"Prediction distribution: {model_predictions.value_counts().to_dict()}")
            logger.info(f"Probabilities generated. Shape: {model_probabilities.shape}")

    except Exception as e:
        logger.critical(f"Error during data cleaning or prediction generation: {e}", exc_info=True)
        sys.exit(1)


    # --- Run Backtest Simulation ---
    logger.info("Initializing and running backtest simulation...")
    try:
        # Pass the original `backtest_data` as the main data source for BacktestSimulator
        # This ensures all original columns (OHLCV, etc.) are available,
        # and predictions/probabilities are correctly aligned inside BacktestSimulator's _prepare_data.
        # Pass app_config.strategy and app_config.backtest directly
        backtest_simulator = BacktestSimulator( # Renamed variable
            data=backtest_data.copy(),
            model_predict=model_predictions.copy(),
            model_proba=model_probabilities.copy(),
            symbol=symbol,
            interval=interval,
            model_type=model_key,
            # No longer passing config objects as arguments here, as they are accessed via app_config internally in Simulator
            # Ensure paths_override is passed if relevant, otherwise omit.
            # backtest_config_obj=app_config.backtest,
            # strategy_config_obj=app_config.strategy,
            # exchange_config_obj=app_config.exchange,
            # feature_config_obj=app_config.features
        )

        trades_df, equity_curve_df, summary_metrics = backtest_simulator.run_backtest() # Call the new simulator

        logger.info("Backtest simulation completed.")
        logger.info(f"Number of trades: {summary_metrics.get('total_trades', 'N/A')}")
        final_capital_val = summary_metrics.get('final_capital', 'N/A') # Use 'final_capital' from simulator's metrics
        if isinstance(final_capital_val, (int, float)):
            logger.info(f"Final Capital: {final_capital_val:.2f}")
        else:
            logger.info(f"Final Capital: {final_capital_val}")


    except Exception as e:
        logger.critical(f"Error during backtest simulation: {e}", exc_info=True)
        sys.exit(1)

    # --- Run Results Analysis (if ResultAnalyser is available) ---
    if RESULTS_ANALYSER_AVAILABLE:
        logger.info("Running results analysis...")
        try:
            # Note: ResultAnalyser expects trades and equity curve to be saved to disk
            # by the simulator itself if save_trades/save_equity_curve were True.
            # The ResultAnalyser then loads them.
            analyser = ResultAnalyser( # Updated class name
                symbol=symbol,
                interval=interval,
                model_type=model_key,
                results_type='backtest', # This now defaults to 'backtest' in ResultAnalyser, but explicit is fine
                # Ensure result_dir and analysis_dir are correct.
                # The simulator should save results to app_config.paths.backtesting_results_dir
                # ResultAnalyser needs to know where to load them from and where to save its own analysis.
                results_dir=PATHS['backtesting_results_dir'],
                analysis_dir=PATHS['backtesting_analysis_dir'] / model_key, # Assuming analysis goes into a subfolder per model
                paths=PATHS
            )
            analyser.run_analysis()
            logger.info(f"Results analysis completed. Outputs saved in {PATHS['backtesting_analysis_dir'] / model_key}")
        except FileNotFoundError as e:
             logger.error(f"ResultAnalyser failed: Could not find backtest result files. Error: {e}")
             logger.warning("Skipping detailed results analysis.")
        except Exception as e:
            logger.error(f"Error during results analysis: {e}", exc_info=True)
            logger.warning("Skipping detailed results analysis due to error.")
    else:
        logger.info("ResultAnalyser not available. Skipping detailed results analysis.")


    # --- Log End of Run ---
    run_timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"--- Backtest Run Complete ({run_timestamp_end}) ---")
    logger.info("-" * 40)


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run backtests on a trained model using historical data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Time interval for candles (e.g., 5m, 1h, 1d)'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(app_config.model.AVAILABLE_MODEL_TYPES.keys()),
        help=f"Model type key from app_config.model.AVAILABLE_MODEL_TYPES to backtest."
    )
    parser.add_argument(
        '--backtest_mode',
        type=str,
        default=app_config.backtest.backtest_mode, # Get default from app_config.backtest
        choices=['full', 'train', 'test'],
        help='Data split to use for backtesting.'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=app_config.model.train_test_split_ratio, # Get default from app_config.model
        help='Fraction of data used for training (determines test set start).'
    )

    args = parser.parse_args()

    if not (0 < args.train_ratio < 1):
        logger.error(f"Invalid --train_ratio: {args.train_ratio}. Must be between 0.0 and 1.0 (exclusive).")
        sys.exit(1)

    try:
        run_backtest_pipeline(
            symbol=args.symbol.upper(),
            interval=args.interval,
            model_key=args.model,
            backtest_mode=args.backtest_mode,
            train_ratio=args.train_ratio
        )
    except SystemExit:
        logger.info("Backtest script finished as requested (SystemExit).")
    except Exception as e:
        logger.critical(f"Unhandled exception occurred in main execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.shutdown()
