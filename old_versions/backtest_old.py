#!/usr/bin/env python3
"""
backtest.py

Orchestrates the backtesting process for a trained trading model.
Simplified version focusing on core steps.

Steps:
1. Parses command-line arguments.
2. Sets up logging.
3. Loads processed data using DataManager.
4. Splits data based on backtest mode.
5. Loads the trained model using ModelTrainer (which uses DataManager).
6. **Cleans backtest_data to remove rows with NaNs in relevant features.**
7. Generates predictions AND probability scores using the loaded model on cleaned data.
8. Initializes and runs the Backtester simulation, passing predictions and probabilities.
9. Runs the ResultsAnalyser.
10. Saves backtest results using the Backtester's internal saving mechanism.

**MODIFIED (NEW)**: Added generation and passing of model probability scores to Backtester.
**FIXED**: Corrected the KeyError by using the correct path key 'processed_data_pattern' from paths.py.
**FIXED**: Corrected the KeyError by using the correct path key 'backtesting_analysis_results_dir' from paths.py when calling ResultsAnalyser.
**FIXED**: Removed the erroneous 'log_dir' argument from the setup_rotating_logging call.
**FIXED**: Corrected argument names when initializing the Backtester to match backtester.py signature.
**MODIFIED**: Switched data loading to use DataManager.load_data with data_type='processed'.
**MODIFIED**: Switched model loading to use ModelTrainer.load, which now internally uses DataManager.load_model_artifact.
**MODIFIED**: Removed manual path construction for loading data and models, relying on DataManager/ModelTrainer.
**MODIFIED**: Ensured predictions are aligned to the backtest data index and NaNs are filled.
**MODIFIED**: Added more robust error handling and logging throughout the pipeline.
**MODIFIED**: Ensured Backtester receives necessary config overrides and paths.
**MODIFIED**: Added check for empty backtest data after splitting.
**MODIFIED**: Standardized symbol to uppercase before passing to pipeline.
**MODIFIED**: Added logging.shutdown() in finally block for clean exit.
**MODIFIED (NEW)**: Added a cleaning step for `backtest_data` to remove rows with NaNs
                 in feature columns used by the model, before making predictions.
                 This aims to prevent NaN/Inf errors from the preprocessor.
**FIXED**: Passed model_key to ResultsAnalyser's analysis_dir to ensure model-specific output directories.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import copy # Import copy for deep copying config


# --- Determine Project Root ---
# Assume this script is in a subdirectory (like 'scripts/') and its parent is the project root.
# This logic is kept for reliable project root detection.
try:
    script_dir = Path(__file__).resolve().parent
    PROJECT_ROOT = script_dir.parent
except NameError:
    # Fallback if __file__ is not defined (e.g., in interactive environments)
    PROJECT_ROOT = Path('.').resolve()
    print(f"Warning: __file__ not defined. Assuming project root is current directory: {PROJECT_ROOT}", file=sys.stderr)

# Add project root to sys.path BEFORE attempting project imports
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import Project Modules ---
# Essential imports for backtesting pipeline
try:
    # Import configuration parameters
    from config.params import MODEL_CONFIG, GENERAL_CONFIG, STRATEGY_CONFIG, BACKTESTER_CONFIG

    # Import centralized path configurations
    from config.paths import PATHS

    # Import the DataManager
    from utils.data_manager import DataManager
    # Import the ModelTrainer
    from utils.model_trainer import ModelTrainer # Import the latest ModelTrainer class

    # Import the Backtester
    from utils.backtester import Backtester # Import the latest Backtester class

    # Import the logging setup function
    from utils.logger_config import setup_rotating_logging

    # Import the ResultsAnalyser (assuming it exists and is needed)
    # Wrap in try-except in case ResultsAnalyser is not yet implemented
    try:
        from utils.results_analyzer import ResultsAnalyser # Assume this exists
        RESULTS_ANALYSER_AVAILABLE = True
    except ImportError:
        RESULTS_ANALYSER_AVAILABLE = False
        print("WARNING: ResultsAnalyser class not found in utils. Skipping detailed analysis.", file=sys.stderr)


except ImportError as e:
    print(f"ERROR: Failed to import necessary project modules: {e}", file=sys.stderr)
    print("Please ensure config/, utils/ directories exist within the project root and contain the required files.", file=sys.stderr)
    sys.exit(1)
except FileNotFoundError as e:
    print(f"ERROR: Configuration file not found: {e}. Ensure config/params.py and config/paths.py exist.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during initial imports: {e}", file=sys.stderr)
    sys.exit(1)


# --- Global Logger Setup ---
# Set up logging specifically for this script using the utility function
# This should be called early to configure logging before extensive logging occurs.
setup_rotating_logging("backtest") # Pass "backtest" as the positional argument
logger = logging.getLogger(__name__) # Re-get logger to use the configured handlers


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
    # --- Log Start of Run ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n--- Starting Backtest Run ({run_timestamp}) ---")
    logger.info(f"Symbol: {symbol.upper()}, Interval: {interval}, Model: {model_key}")
    logger.info(f"Backtest Mode: {backtest_mode}, Train Ratio: {train_ratio}")
    logger.info("-" * 40)

    dm = DataManager() # Initialize DataManager for data loading

    # --- Load Processed Data using DataManager ---
    logger.info("Loading processed data using DataManager...")
    try:
        # Use the new DataManager.load_data(symbol, interval, data_type)
        data = dm.load_data(symbol=symbol, interval=interval, data_type='processed')

        # Basic data validation and index handling
        if data is None or data.empty:
            raise FileNotFoundError(f"Processed data is empty or could not be loaded for {symbol} {interval}.")

        # Ensure index is DatetimeIndex and UTC timezone-aware
        if not isinstance(data.index, pd.DatetimeIndex):
             logger.warning("Data index is not DatetimeIndex. Attempting conversion.")
             data.index = pd.to_datetime(data.index, utc=True)
        elif data.index.tz is None:
             logger.warning("Data index is timezone naive. Assuming UTC.")
             data.index = data.index.tz_localize('UTC')
        elif str(data.index.tz) != 'UTC':
             logger.warning(f"Data index has timezone {data.index.tz}. Converting to UTC.")
             data.index = data.index.tz_convert('UTC')

        # Ensure index is monotonic increasing
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
    original_backtest_data_len = 0 # To track rows before cleaning
    try:
        if backtest_mode == 'full':
            backtest_data = data.copy()
        elif backtest_mode in ['train', 'test']:
            # Ensure train_ratio is valid (checked in main, but defensive here)
            if not (0 < train_ratio < 1):
                 raise ValueError(f"Invalid train_ratio: {train_ratio}. Must be between 0 and 1 (exclusive).")

            train_size = int(len(data) * train_ratio)
            if train_size <= 0 or train_size >= len(data):
                raise ValueError(f"Invalid train_ratio ({train_ratio}) results in empty train or test set for split size {train_size} from total {len(data)}.")

            if backtest_mode == 'train':
                backtest_data = data.iloc[:train_size].copy()
            else: # test mode
                backtest_data = data.iloc[train_size:].copy()
        else:
            # This should be caught by argparse choices, but defensive check
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
    try:
        # Get model-specific config and pass to ModelTrainer
        model_specific_config = MODEL_CONFIG.get(model_key)
        if model_specific_config is None:
             raise ValueError(f"Model configuration for '{model_key}' not found in MODEL_CONFIG.")

        trainer = ModelTrainer(config=model_specific_config.copy()) # Pass a copy of the config
        trainer.load(symbol=symbol, interval=interval, model_key=model_key)

        if not hasattr(trainer, 'model') and not hasattr(trainer, 'pipeline'): # Check for model OR pipeline
             raise RuntimeError("ModelTrainer failed to load model or pipeline.")
        if not hasattr(trainer, 'preprocessor') or trainer.preprocessor is None:
            raise RuntimeError("ModelTrainer failed to load the preprocessor.")
        if not hasattr(trainer, 'feature_columns_original') or not trainer.feature_columns_original:
            logger.warning("Original feature columns not found in loaded model metadata. This might affect data cleaning for predictions.")
            # Fallback: attempt to use features_to_use from config if original_columns are missing
            # This is a less ideal fallback, as metadata should be the source of truth.
            trainer.feature_columns_original = model_specific_config.get('features_to_use')
            if not trainer.feature_columns_original: # If still none, this is a problem
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
    # This step is crucial to mimic the cleaning done during training.
    logger.info("Cleaning backtest_data features before prediction...")
    try:
        # Get the feature columns the model was trained on (original names before preprocessing)
        # These should be stored in trainer.feature_columns_original by the ModelTrainer.load method
        model_feature_cols = trainer.feature_columns_original
        if not model_feature_cols:
            raise ValueError("Original feature columns used by the model are not available from the loaded trainer. Cannot clean data for prediction.")

        # Ensure all expected feature columns are present in backtest_data
        missing_cols_in_backtest_data = [col for col in model_feature_cols if col not in backtest_data.columns]
        if missing_cols_in_backtest_data:
            logger.error(f"Backtest data is missing expected feature columns: {missing_cols_in_backtest_data}")
            # If critical features are missing, predictions will likely fail or be inaccurate.
            # Depending on the model, it might be okay if some *optional* features are missing,
            # but the preprocessor expects all columns it was fitted on.
            # For now, log an error and proceed, but this is a potential issue.
            # Consider raising an error if this list is not empty.
            # For safety, let's filter model_feature_cols to only those present in backtest_data
            model_feature_cols = [col for col in model_feature_cols if col in backtest_data.columns]
            if not model_feature_cols:
                 raise ValueError("No usable feature columns found in backtest_data that match model's expected features.")


        # Drop rows with NaNs in any of the model's feature columns
        # This mirrors the cleaning done in train_model.py before splitting.
        backtest_data_cleaned = backtest_data.dropna(subset=model_feature_cols).copy()

        rows_removed_cleaning = original_backtest_data_len - len(backtest_data_cleaned)
        if rows_removed_cleaning > 0:
            logger.info(f"Removed {rows_removed_cleaning} rows from backtest_data due to NaNs in model feature columns: {model_feature_cols}")

        if backtest_data_cleaned.empty:
            logger.warning("Backtest data is empty after cleaning NaNs in feature columns. No predictions will be made.")
            # If data is empty, create empty predictions and probabilities and skip to results
            model_predictions = pd.Series(dtype=np.int8, index=pd.DatetimeIndex([]))
            model_probabilities = pd.DataFrame(dtype=float, index=pd.DatetimeIndex([]))
        else:
            logger.info(f"Backtest data cleaned. Shape after cleaning: {backtest_data_cleaned.shape}")
            # Update backtest_data to the cleaned version
            backtest_data = backtest_data_cleaned

    except Exception as e:
        logger.critical(f"Error cleaning backtest_data features: {e}", exc_info=True)
        logger.warning("Proceeding with uncleaned backtest_data. Predictions may fail or be inaccurate.")
        # If cleaning fails, proceed with original backtest_data, but log a strong warning.
        # The NaN/Inf error might still occur in trainer.predict()/predict_proba().


    # --- Generate Predictions and Probabilities ---
    if not backtest_data.empty: # Only predict if there's data left after cleaning
        logger.info("Generating predictions and probability scores using the loaded model on cleaned data...")
        try:
            # trainer.predict handles feature selection (based on loaded metadata), scaling, and LSTM sequencing.
            # It should return a pandas Series aligned to the input data's index (or a subset for LSTM)
            # Pass the potentially cleaned backtest_data DataFrame
            model_predictions = trainer.predict(backtest_data) # Use the cleaned backtest_data

            # Get probability predictions
            model_probabilities = trainer.predict_proba(backtest_data)

            # Basic validation and cleanup of predictions
            if not isinstance(model_predictions, pd.Series):
                raise TypeError(f"Model prediction output is not a pandas Series (type: {type(model_predictions)}).")

            # Ensure predictions are aligned to the *cleaned* backtest_data index.
            # If trainer.predict() already aligns to its input (cleaned backtest_data), this might be redundant,
            # but it's a good safeguard.
            if not model_predictions.index.equals(backtest_data.index):
                 logger.warning("Prediction index does not match cleaned backtest data index. Reindexing and filling missing with 0.")
                 model_predictions = model_predictions.reindex(backtest_data.index) # Reindex to cleaned data

            # Fill potential NaNs in predictions (can happen with LSTM for initial bars or reindexing)
            if model_predictions.isnull().any():
                 logger.warning("NaN values found in predictions after alignment. Filling with 0 (neutral).")
                 model_predictions.fillna(0, inplace=True)

            # Ensure predictions are in the expected range [-1, 0, 1] and are integer type
            model_predictions = model_predictions.clip(-1, 1).astype(int)
            if not np.all(model_predictions.isin([-1, 0, 1])):
                 logger.warning("Predictions contain values other than -1, 0, 1 after clipping. This is unexpected.")


            # Basic validation and cleanup of probabilities
            if model_probabilities is None or model_probabilities.empty:
                 logger.warning("Model predict_proba returned None or empty DataFrame. Confidence scores will not be used.")
                 # Create an empty DataFrame with the correct index if predict_proba failed
                 model_probabilities = pd.DataFrame(dtype=float, index=backtest_data.index)
            else:
                 if not isinstance(model_probabilities, pd.DataFrame):
                      raise TypeError(f"Model probability output is not a pandas DataFrame (type: {type(model_probabilities)}).")

                 # Ensure probabilities are aligned to the *cleaned* backtest_data index.
                 if not model_probabilities.index.equals(backtest_data.index):
                      logger.warning("Probability index does not match cleaned backtest data index. Reindexing and filling missing with NaN.")
                      # Reindex to cleaned data, fill missing rows with NaN
                      model_probabilities = model_probabilities.reindex(backtest_data.index)

                 # Check expected columns (-1, 0, 1)
                 expected_proba_cols = [-1, 0, 1]
                 if not all(col in model_probabilities.columns for col in expected_proba_cols):
                      logger.warning(f"Probability DataFrame is missing expected columns ({expected_proba_cols}). Found: {model_probabilities.columns.tolist()}. Confidence scores may be unreliable.")
                      # Fill missing columns with NaN
                      for col in expected_proba_cols:
                           if col not in model_probabilities.columns:
                                model_probabilities[col] = np.nan

                 # Ensure probabilities sum to approximately 1 across rows (ignoring NaNs)
                 row_sums = model_probabilities[expected_proba_cols].sum(axis=1, skipna=True)
                 # Use tolerance for float comparison
                 if not np.allclose(row_sums[row_sums.notna()], 1.0, atol=1e-6):
                      logger.warning("Probability row sums do not equal 1.0 for some rows. Probabilities may be inaccurate.")

                 # Fill potential NaNs in probabilities (can happen with LSTM for initial bars or reindexing)
                 if model_probabilities.isnull().any().any():
                      logger.warning("NaN values found in probabilities after alignment. Filling with NaN.")
                      # Filling with NaN is appropriate here, as missing probability means no confidence info.
                      # The Backtester will need to handle these NaNs.


            logger.info(f"Predictions generated. Shape: {model_predictions.shape}")
            logger.debug(f"Prediction distribution: {model_predictions.value_counts().to_dict()}")
            logger.info(f"Probabilities generated. Shape: {model_probabilities.shape}")

        except Exception as e:
            logger.critical(f"Error generating predictions or probabilities: {e}", exc_info=True)
            sys.exit(1)
    else: # If backtest_data became empty after cleaning
        logger.info("Skipping prediction and probability generation as backtest_data is empty after cleaning.")
        # Ensure model_predictions and model_probabilities are empty Series/DataFrame with correct types if no predictions are made
        # Use the original data's index type if available, otherwise default
        original_index_for_empty = data.index if not data.empty else pd.DatetimeIndex([])
        model_predictions = pd.Series(dtype=np.int8, index=original_index_for_empty)
        model_probabilities = pd.DataFrame(dtype=float, index=original_index_for_empty)


    # --- Run Backtest Simulation ---
    logger.info("Initializing and running backtest simulation...")
    try:
        # Pass the potentially cleaned backtest_data (including OHLCV and any other columns needed by Backtester)
        # Pass the aligned model_predictions Series
        # Pass the aligned model_probabilities DataFrame (NEW)
        # Pass the relevant configurations directly, using the correct parameter names
        # Use copies of configs to ensure Backtester doesn't modify original dictionaries
        backtester = Backtester(
            data=backtest_data.copy(), # Pass a copy of the (potentially cleaned) data
            model_predict=model_predictions.copy(), # Pass a copy of predictions
            model_proba=model_probabilities.copy(), # Pass a copy of probabilities (NEW)
            symbol=symbol,
            interval=interval,
            model_type=model_key,
            backtester_config_override=copy.deepcopy(BACKTESTER_CONFIG), # Pass deep copies
            strategy_config_override=copy.deepcopy(STRATEGY_CONFIG),
            paths_override=copy.deepcopy(PATHS) # Pass a deep copy of paths
        )

        # Run the backtest - results are saved internally by Backtester
        trades_df, equity_curve_df, summary_metrics = backtester.run_backtest()

        logger.info("Backtest simulation completed.")
        logger.info(f"Number of trades: {summary_metrics.get('num_trades', 'N/A')}")
        # Ensure final_equity is formatted correctly, handling potential non-float values
        final_equity_val = summary_metrics.get('final_equity', 'N/A')
        if isinstance(final_equity_val, (int, float)):
            logger.info(f"Final Equity: {final_equity_val:.2f}")
        else:
            logger.info(f"Final Equity: {final_equity_val}")


    except Exception as e:
        logger.critical(f"Error during backtest simulation: {e}", exc_info=True)
        sys.exit(1)

    # --- Run Results Analysis (if ResultsAnalyser is available) ---
    if RESULTS_ANALYSER_AVAILABLE:
        logger.info("Running results analysis...")
        try:
            # Instantiate ResultsAnalyser - it will load the saved results files
            analyser = ResultsAnalyser(
                symbol=symbol,
                interval=interval,
                model_type=model_key,
                results_type='backtest', # Specify analysis type
                results_dir=PATHS['backtesting_results_dir'],
                # Pass the model-specific analysis directory
                analysis_dir=PATHS['backtesting_analysis_dir'] / model_key, # FIXED: Model-specific analysis directory
                paths=PATHS # Pass the full PATHS dictionary
            )
            analyser.run_analysis()
            logger.info(f"Results analysis completed. Outputs saved in {PATHS['backtesting_analysis_dir'] / model_key}") # Log the correct path
        except FileNotFoundError as e:
             logger.error(f"ResultsAnalyser failed: Could not find backtest result files. Error: {e}")
             logger.warning("Skipping detailed results analysis.")
        except Exception as e:
            logger.error(f"Error during results analysis: {e}", exc_info=True)
            logger.warning("Skipping detailed results analysis due to error.")
    else:
        logger.info("ResultsAnalyser not available. Skipping detailed results analysis.")


    # --- Log End of Run ---
    run_timestamp_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"--- Backtest Run Complete ({run_timestamp_end}) ---")
    logger.info("-" * 40)


# --- Script Entry Point ---
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run backtests on a trained model using historical data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
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
        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'], # Match fetch_data choices
        help='Time interval for candles (e.g., 5m, 1h, 1d)'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=list(MODEL_CONFIG.keys()), # Use keys from MODEL_CONFIG
        help=f"Model type key from MODEL_CONFIG to backtest."
    )
    parser.add_argument(
        '--backtest_mode',
        type=str,
        default='test',
        choices=['full', 'train', 'test'],
        help='Data split to use for backtesting.'
    )
    # Use GENERAL_CONFIG for the default train_ratio
    default_train_ratio = GENERAL_CONFIG.get('train_test_split_ratio', 0.8) # Use train_test_split_ratio key
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=default_train_ratio,
        help='Fraction of data used for training (determines test set start).'
    )

    args = parser.parse_args()

    # Validate train_ratio
    if not (0 < args.train_ratio < 1):
        logger.error(f"Invalid --train_ratio: {args.train_ratio}. Must be between 0.0 and 1.0 (exclusive).")
        sys.exit(1)

    # --- Run Pipeline ---
    try:
        # Standardize symbol to uppercase before passing to pipeline
        run_backtest_pipeline(
            symbol=args.symbol.upper(),
            interval=args.interval,
            model_key=args.model,
            backtest_mode=args.backtest_mode,
            train_ratio=args.train_ratio
        )
    except SystemExit:
        # Catch SystemExit to prevent traceback on intentional sys.exit() calls
        logger.info("Backtest script finished as requested (SystemExit).")
    except Exception as e:
        # Catch any other unhandled exceptions and log them
        logger.critical(f"Unhandled exception occurred in main execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Ensure all log handlers are closed properly on script exit
        logging.shutdown()


    """
    Usage example:

    Run backtest on the test set for the trained XGBoost model for BTCUSDT 1h data:
        python scripts/backtest.py --symbol BTCUSDT --interval 1h --model xgboost

    Run backtest on the test set for the trained RandomForest model for ADAUSDT 5m data:
        python scripts/backtest.py --symbol ADAUSDT --interval 5m --model random_forest

    Run backtest on the full dataset for the trained LSTM model for ADAUSDT 5m data:
        python scripts/backtest.py --symbol ADAUSDT --interval 5m --model lstm --backtest_mode full --train_ratio 0.7

    Ensure you have processed data files in your data/processed directory,
    labeled data files (containing labels) in your data/labeled directory,
    and trained model files in your models/trained_models directory.
    config/params.py (with MODEL_CONFIG, GENERAL_CONFIG, STRATEGY_CONFIG, BACKTESTER_CONFIG)
    and config/paths.py are correctly configured.
    """
