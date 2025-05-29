# scripts/convert_trades_to_json.py

import pandas as pd
import json
import sys
import logging
from pathlib import Path
import argparse
import numpy as np
import math
from typing import Optional # Import Optional for type hinting

# Add the project root to the system path to allow importing modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import paths from your project's config
try:
    from config.paths import PATHS
except ImportError:
    print("ERROR: Failed to import PATHS from config.paths.", file=sys.stderr)
    print("Ensure config/paths.py exists and is accessible.")
    sys.exit(1)

# Import the DataManager utility (still needed for loading OHLCV)
try:
    from utils.data_manager import DataManager
except ImportError:
    print("ERROR: Failed to import DataManager from utils.data_manager.", file=sys.stderr)
    print("Ensure utils/data_manager.py exists and is accessible.")
    sys.exit(1)

# Import the rotating logger setup
try:
    from utils.logger_config import setup_rotating_logging
except ImportError:
    print("ERROR: Failed to import setup_rotating_logging from utils.logger_config.", file=sys.stderr)
    print("Ensure utils/logger_config.py exists and is accessible.")
    sys.exit(1)


# Set up logging for this script using the rotating logger configuration
# The log file will be named 'convert_trades_to_json.log'
logger = setup_rotating_logging(
    log_filename_base=Path(__file__).stem, # Use the script's filename as the base for the log file
    log_level=logging.INFO # Changed from DEBUG to INFO to reduce verbosity
)

# Define a small epsilon for float comparisons
FLOAT_EPSILON = 1e-6


def handle_value(value):
    """
    Helper function to handle different data types for JSON serialization,
    converting Timestamps to milliseconds since epoch and handling NaN/Infinity values.
    """
    if pd.isna(value):
        logger.debug(f"handle_value: Input is NaN/NaT, returning None. Value: {value}, Type: {type(value)}")
        return None # Convert pandas NaN to JSON null

    if isinstance(value, pd.Timestamp):
        # Convert pandas Timestamp to milliseconds since epoch (required by Lightweight Charts)
        # Ensure timezone-aware timestamp before converting to epoch
        if value.tzinfo is None:
             # Assume UTC if timezone is not specified (adjust if your data uses a different timezone)
             value = value.tz_localize('UTC')
        # Convert timestamp to milliseconds
        milliseconds = int(value.timestamp() * 1000)
        logger.debug(f"handle_value: Converted Timestamp {value} to milliseconds {milliseconds}.")
        return milliseconds

    # Handle numeric types, including numpy numeric types
    if isinstance(value, (int, float, np.integer, np.floating)):
        python_value = value.item() if isinstance(value, (np.integer, np.floating)) else value
        if math.isinf(python_value) or math.isnan(python_value):
            logger.debug(f"handle_value: Input is numeric NaN/Inf, returning None. Value: {python_value}")
            return None
        logger.debug(f"handle_value: Returning numeric value {python_value}.")
        return python_value

    # Handle boolean values explicitly, as some versions of pandas might not serialize them directly
    if isinstance(value, bool):
        logger.debug(f"handle_value: Returning boolean value {value}.")
        return bool(value)

    logger.debug(f"handle_value: Unhandled type {type(value)}, returning original value {value}.")
    return value


def convert_trade_history_and_ohlcv_to_json(
    ohlcv_df: pd.DataFrame,
    trade_history_df: pd.DataFrame,
    output_file_path: Path
):
    """
    Uses loaded OHLCV and trade history data, combines them, and saves as a JSON file
    structured for financial chart visualization.

    Args:
        ohlcv_df (pd.DataFrame): Loaded OHLCV data DataFrame.
        trade_history_df (pd.DataFrame): Loaded trade history DataFrame.
        output_file_path (Path): Path where the output JSON file will be saved.
    """
    logger.debug(f"Input ohlcv_df received by convert_trade_history_and_ohlcv_to_json. Shape: {ohlcv_df.shape}")
    logger.debug(f"Input ohlcv_df columns: {ohlcv_df.columns.tolist()}")
    logger.debug(f"Input ohlcv_df dtypes:\n{ohlcv_df.dtypes}")
    logger.debug(f"Input ohlcv_df head:\n{ohlcv_df.head()}")


    # Select and rename OHLCV columns for Lightweight Charts format
    required_ohlcv_cols = ['open', 'high', 'low', 'close']
    # IMPORTANT: Ensure 'open_time' is also present and used for the 'time' column
    if not all(col in ohlcv_df.columns for col in required_ohlcv_cols + ['open_time']):
         missing = [col for col in required_ohlcv_cols + ['open_time'] if col not in ohlcv_df.columns]
         logger.error(f"Loaded OHLCV data is missing required columns: {missing}. Aborting.")
         sys.exit(1)

    # Create a new DataFrame with only the required OHLC columns and ensure they are numeric
    ohlcv_data = ohlcv_df[required_ohlcv_cols].astype(float).copy()
    ohlcv_data.columns = ['open', 'high', 'low', 'close'] # Ensure lowercase names after selection

    # FIX: Explicitly ensure 'open_time' is datetime and then convert to milliseconds
    # This is the most critical part to ensure correct timestamps for the chart.
    if not pd.api.types.is_datetime64_any_dtype(ohlcv_df['open_time']):
        logger.warning("ohlcv_df['open_time'] is not datetime type. Attempting conversion.")
        ohlcv_df['open_time'] = pd.to_datetime(ohlcv_df['open_time'], errors='coerce', utc=True)
        # Drop rows where conversion might have failed (resulted in NaT)
        ohlcv_df.dropna(subset=['open_time'], inplace=True)

    # Use the 'open_time' column for the 'time' property, converting to milliseconds
    # This should now reliably use the correct timestamps from the 'open_time' column.
    ohlcv_data['time'] = (ohlcv_df['open_time'].astype(np.int64) // 10**6)


    # --- NEW DEBUGGING LOGGING for OHLCV data before JSON conversion ---
    logger.debug(f"OHLCV data DataFrame BEFORE JSON conversion:")
    logger.debug(f"Columns: {ohlcv_data.columns.tolist()}")
    logger.debug(f"Data Types:\n{ohlcv_data.dtypes}")
    logger.debug(f"First 5 rows of OHLCV data:\n{ohlcv_data.head()}")
    logger.debug(f"Any NaNs in OHLCV data (open/high/low/close)?\n{ohlcv_data[required_ohlcv_cols].isnull().sum()}")
    logger.debug(f"Any Infs in OHLCV data (open/high/low/close)?\n{np.isinf(ohlcv_data[required_ohlcv_cols].values).sum(axis=0)}")
    logger.debug(f"First 5 values of 'time' column (milliseconds): {ohlcv_data['time'].head().tolist()}")
    # --- END NEW DEBUGGING LOGGING ---


    # --- Prepare Trade Data ---
    trade_markers = []
    if not trade_history_df.empty:
        # Ensure time columns are datetime and UTC before processing
        for col in ['entry_time', 'exit_time']:
            if col in trade_history_df.columns:
                 # Convert to datetime, coercing errors to NaT, ensure UTC
                 try:
                      trade_history_df[col] = pd.to_datetime(trade_history_df[col], errors='coerce', utc=True)
                 except Exception as e:
                      logger.warning(f"Could not convert '{col}' to datetime: {e}. Rows with invalid times will be dropped.")
                      trade_history_df[col] = pd.NaT # Set to NaT on error

        # Drop rows where critical time columns are NaT after conversion attempt
        initial_rows = len(trade_history_df)
        trade_history_df.dropna(subset=['entry_time', 'exit_time'], inplace=True)
        if len(trade_history_df) < initial_rows:
             logger.warning(f"Dropped {initial_rows - len(trade_history_df)} rows from trade history due to invalid time data.")


        # Only proceed if trade_history_df is not empty after checks
        if not trade_history_df.empty:
            # Sort trades by entry time
            trade_history_df = trade_history_df.sort_values(by='entry_time').reset_index(drop=True)


            for index, trade in trade_history_df.iterrows():
                # --- NEW DEBUGGING LOGGING FOR TRADE TIMES ---
                logger.debug(f"Trade {index}: Original entry_time: {trade.get('entry_time')} (Type: {type(trade.get('entry_time'))})")
                logger.debug(f"Trade {index}: Original exit_time: {trade.get('exit_time')} (Type: {type(trade.get('exit_time'))})")
                # --- END NEW DEBUGGING LOGGING ---

                # Use .get() with a default of None to safely access columns that might be missing
                entry_time_ms = handle_value(trade.get('entry_time'))
                exit_time_ms = handle_value(trade.get('exit_time'))

                # --- NEW DEBUGGING LOGGING AFTER handle_value ---
                logger.debug(f"Trade {index}: Processed entry_time_ms: {entry_time_ms} (Type: {type(entry_time_ms)})")
                logger.debug(f"Trade {index}: Processed exit_time_ms: {exit_time_ms} (Type: {type(exit_time_ms)})")
                # --- END NEW DEBUGGING LOGGING ---

                # Ensure 'direction' is retrieved before skipping
                direction = trade.get('direction')

                # Skip trades with invalid times or direction
                if entry_time_ms is None or exit_time_ms is None or direction is None:
                     logger.warning(f"Skipping trade {index} due to missing or invalid time or direction data (entry_time_ms={entry_time_ms}, exit_time_ms={exit_time_ms}, direction={direction}).")
                     continue

                # Ensure direction is treated as a number for comparison
                try:
                    direction_numeric = float(direction)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping trade {index} due to invalid direction value: {direction}")
                    continue

                # Retrieve other necessary trade details for marker text/details
                net_pnl = trade.get('net_pnl')
                exit_reason = trade.get('exit_reason')
                entry_price = trade.get('entry_price')
                exit_price = trade.get('exit_price')


                # Create markers for entry and exit
                # Entry Marker
                trade_markers.append({
                    'time': entry_time_ms, # Use time in milliseconds
                    'position': 'belowBar' if direction_numeric > 0 else 'aboveBar', # Below bar for long entry, above for short entry
                    'color': '#26A69A' if direction_numeric > 0 else '#EF5350', # Green for long, red for short
                    'shape': 'arrowUp' if direction_numeric > 0 else 'arrowDown', # Up arrow for long entry, down for short entry
                    'text': f'Entry ({direction_numeric > 0 and "Long" or "Short"}): {entry_price:.6f}' if entry_price is not None else f'Entry ({direction_numeric > 0 and "Long" or "Short"})',
                    'size': 1.5,
                     # Store full details, ensuring values are handled for JSON
                     'tradeDetails': {k: handle_value(v) for k, v in trade.items()} # handle_value converts timestamps to milliseconds
                })

                # Exit Marker (Plot at exit time)
                # Check if exit time is valid before adding exit marker
                if exit_time_ms is not None:
                    trade_markers.append({
                        'time': exit_time_ms, # Use time in milliseconds
                        'position': 'aboveBar' if direction_numeric > 0 else 'belowBar', # Above bar for long exit, below for short exit
                        'color': '#26A69A' if (net_pnl is not None and net_pnl is not None and net_pnl >= 0) else '#EF5350', # Green if profitable, red if loss
                        'shape': 'circle', # Circle for exit
                        'text': f'Exit ({exit_reason}): {exit_price:.6f} PnL: {net_pnl:.2f}' if exit_price is not None and net_pnl is not None else f'Exit ({exit_reason})',
                        'size': 1.5,
                         # Store full details, ensuring values are handled for JSON
                         'tradeDetails': {k: handle_value(v) for k, v in trade.items()} # handle_value converts timestamps to milliseconds
                    })
                else:
                    logger.warning(f"Trade {index} has invalid exit_time. Skipping exit marker.")

    else:
        logger.warning("Trade history is empty. No trade markers will be generated.")

    # --- Combine Data for JSON Output ---
    output_data = {
        # IMPORTANT: ohlcv_data should already be correctly formatted with 'time', 'open', 'high', 'low', 'close'
        # as numeric values and 'time' in milliseconds.
        'ohlcv': ohlcv_data.to_dict(orient='records'), # No need to reset_index(drop=True) if index is already handled
        'tradeMarkers': trade_markers
    }

    logger.info(f"Saving combined data to JSON file: {output_file_path}...")
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    logger.info(f"Combined data successfully converted and saved to {output_file_path}")


def main():
    """
    Main function to parse arguments and run the conversion.
    Uses DataManager for loading OHLCV data and paths from config.paths.
    Manually constructs path and loads trade history DataFrame using pandas.
    """
    parser = argparse.ArgumentParser(description="Convert trade history and OHLCV data to JSON for visualization.")
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', required=True, help='Data interval (e.g., 1h, 5m)')
    parser.add_argument('--model_type', required=True, help='Model type (e.g., xgboost, random_forest)')
    parser.add_argument('--results_type', default='backtest', choices=['backtest', 'live'],
                        help='Type of results (backtest or live). Defaults to backtest.')
    parser.add_argument('--output_file', type=Path,
                        help='Optional: Path for the output JSON file. Overrides default path from paths.py.')

    args = parser.parse_args()

    # Instantiate DataManager
    dm = DataManager()

    # --- Load OHLCV data using DataManager ---
    # DataManager.load_data for 'raw' data type is appropriate here
    ohlcv_df = dm.load_data(symbol=args.symbol, interval=args.interval, data_type='raw')

    if ohlcv_df is None or ohlcv_df.empty:
        logger.error("Failed to load OHLCV data using DataManager. Aborting.")
        sys.exit(1)

    # Reset index to make 'open_time' a regular column for processing
    # The DataManager sets 'open_time' as the index, so we need to bring it back as a column.
    if ohlcv_df.index.name == 'timestamp' and 'open_time' not in ohlcv_df.columns:
        ohlcv_df = ohlcv_df.reset_index().rename(columns={'timestamp': 'open_time'})
        logger.debug("Reset index and renamed 'timestamp' to 'open_time' for OHLCV data.")


    # --- NEW DEBUGGING LOGGING for ohlcv_df after loading ---
    logger.debug(f"ohlcv_df loaded from DataManager. Shape: {ohlcv_df.shape}")
    logger.debug(f"ohlcv_df columns: {ohlcv_df.columns.tolist()}")
    logger.debug(f"ohlcv_df dtypes:\n{ohlcv_df.dtypes}")
    logger.debug(f"ohlcv_df head:\n{ohlcv_df.head()}")
    logger.debug(f"Any NaNs in loaded ohlcv_df? {ohlcv_df.isnull().values.any()}")
    logger.debug(f"Any Infs in loaded ohlcv_df? {np.isinf(ohlcv_df.select_dtypes(include=np.number).values).any()}")
    # --- END NEW DEBUGGING LOGGING ---


    # --- Manually Construct Path and Load Trade History Data ---
    # Determine the base results directory based on results_type
    if args.results_type == 'backtest':
        results_dir_key = "backtesting_results_dir"
        # Use the correct, pluralized key name from paths.py
        trades_pattern_key = "backtesting_trades_pattern"
    elif args.results_type == 'live':
        results_dir_key = "live_trading_results_dir"
        # Use the correct, pluralized key name from paths.py
        trades_pattern_key = "live_trading_trades_pattern"
    else: # Should not happen due to argparse choices, but for safety
        logger.error(f"Invalid results_type '{args.results_type}'. Must be 'backtest' or 'live'.")
        sys.exit(1)

    results_dir = PATHS.get(results_dir_key)
    trades_pattern = PATHS.get(trades_pattern_key)

    if not results_dir or not isinstance(results_dir, Path):
         logger.error(f"Missing or invalid results directory key '{results_dir_key}' in paths.py.")
         sys.exit(1)
    if not trades_pattern:
         logger.error(f"Missing trades file pattern key '{trades_pattern_key}' in paths.py.")
         sys.exit(1)


    trade_history_df = pd.DataFrame() # Initialize as empty DataFrame

    try:
        # Manually format the trade history file path using the pattern
        trade_history_file_path = Path(results_dir) / trades_pattern.format(
             symbol=args.symbol, interval=args.interval, model_type=args.model_type)
        logger.info(f"Attempting to load trade history from {trade_history_file_path}")

        if not trade_history_file_path.exists():
            logger.warning(f"Trade history file not found: {trade_history_file_path}. Proceeding without trade markers.")
            # trade_history_df remains empty DataFrame

        else:
            # Load trade history directly using pandas based on file extension
            try:
                if trade_history_file_path.suffix.lower() == '.parquet':
                    trade_history_df = pd.read_parquet(trade_history_file_path)
                elif trade_history_file_path.suffix.lower() == '.csv':
                    # Use keep_default_na=False to correctly load empty strings as empty strings, not NaN
                    trade_history_df = pd.read_csv(trade_history_file_path, keep_default_na=False)
                else:
                    logger.error(f"Unsupported file format for trade history: {trade_history_file_path.suffix}. Please use .parquet or .csv. Proceeding without trade markers.")
                    trade_history_df = pd.DataFrame() # Ensure it's empty on error

                if not trade_history_df.empty:
                    logger.info(f"Trade history loaded successfully. Shape: {trade_history_df.shape}")
                else:
                    logger.warning(f"Trade history file found but is empty: {trade_history_file_path}. Proceeding without trade markers.")


            except Exception as e:
                logger.error(f"Error loading trade history from {trade_history_file_path}: {e}", exc_info=True)
                logger.warning("Proceeding without trade markers due to loading error.")
                trade_history_df = pd.DataFrame() # Use empty DataFrame on error


    except KeyError as e:
        logger.error(f"Missing key in trade history path pattern '{trades_pattern}': {e}")
        logger.warning("Proceeding without trade markers due to path error.")
        trade_history_df = pd.DataFrame() # Use empty DataFrame on error
    except Exception as e:
        logger.error(f"Error constructing trade history file path: {e}")
        logger.warning("Proceeding without trade markers due to path error.")
        trade_history_df = pd.DataFrame() # Use empty DataFrame on error


    # Determine default output file path using the determined results directory
    if args.output_file:
        output_file_path = args.output_file
        logger.info(f"Using specified output file path: {output_file_path}")
    else:
        # Define a pattern for the combined analysis JSON output within the results directory
        combined_analysis_json_pattern = "{symbol}_{interval}_{model_type}_trades_ohlcv_analysis.json"

        try:
            # Save the JSON file directly in the results directory
            output_file_path = Path(results_dir) / combined_analysis_json_pattern.format(
                 symbol=args.symbol, interval=args.interval, model_type=args.model_type)
            logger.info(f"Using default output file path: {output_file_path}")
        except KeyError as e:
            logger.error(f"Missing key in output path pattern '{combined_analysis_json_pattern}': {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error formatting output file path: {e}")
            sys.exit(1)


    # Run the conversion
    convert_trade_history_and_ohlcv_to_json(
        ohlcv_df=ohlcv_df,
        trade_history_df=trade_history_df, # Pass the potentially empty DataFrame
        output_file_path=output_file_path
    )


if __name__ == "__main__":
    main()
