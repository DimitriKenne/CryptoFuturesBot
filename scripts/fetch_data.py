#!/usr/bin/env python3
"""
Script to fetch historical futures market data from Binance
using the BinanceFuturesAdapter and save it using the DataManager.

Loads API keys from environment variables using python-dotenv
and uses the actual BinanceFuturesAdapter.
"""

import argparse
from pathlib import Path
import logging
import sys
import os
from dotenv import load_dotenv
import asyncio # Import asyncio for running async functions
# Import Binance exceptions for specific error handling
from binance.exceptions import BinanceAPIException, BinanceRequestException

# Import necessary types and libraries
from typing import Optional # Import Optional for type hinting
import pandas as pd # Import pandas as pd

# --- Load Environment Variables ---
# This should be one of the first things your script does.
# It looks for a .env file in the current directory or parent directories
# and loads the key-value pairs into the environment.
load_dotenv()

# --- Configuration and Imports ---
# Assuming project root is the parent of the 'scripts' directory
PROJECT_ROOT = Path(__file__).parent.parent
# Add project root to Python path for imports (consider alternatives for larger projects)
sys.path.append(str(PROJECT_ROOT))

# Import configuration parameters
# Assumes config/params.py exists and contains dictionaries like STRATEGY_CONFIG, EXCHANGE_CONFIG
try:
    from config.params import STRATEGY_CONFIG, EXCHANGE_CONFIG
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import configuration modules (config/params.py). "
          f"Ensure config/ is correctly structured. Error: {e}", file=sys.stderr)
    sys.exit(1)


# Import centralized path configurations
# Assumes config/paths.py exists and defines PATHS dictionary
try:
    from config.paths import PATHS
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import configuration modules (config/paths.py). "
          f"Ensure config/ is correctly structured. Error: {e}", file=sys.stderr)
    sys.exit(1)


# Import the logger setup utility
# Assumes it's located in utils/logger_config.py
try:
    from utils.logger_config import setup_rotating_logging
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import logger utility (utils/logger_config.py). Error: {e}", file=sys.stderr)
    sys.exit(1)


# --- Set up Logging ---
# Use the utility function to configure the root logger with rotating file and console handlers.
# The log file will be named 'fetch_data.log' and saved in the directory specified by PATHS['logs_dir'].
try:
    # The setup_rotating_logging function handles getting the log directory from PATHS internally
    setup_rotating_logging('fetch_data')
except Exception as e:
    # Fallback to basic stream logging if rotating setup fails
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(lineno)d]',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    print(f"WARNING: Failed to configure rotating logging: {e}. Using basic stdout logging.", file=sys.stderr)


# Get a logger instance for this script.
# This is good practice even when using the root logger setup.
logger = logging.getLogger(__name__)


try:
    # Import the actual BinanceFuturesAdapter
    # Assumes it's located in adapters/binance_futures_adapter.py
    from adapters.binance_futures_adapter import BinanceFuturesAdapter

    # Import the DataManager
    # Assumes it's located in utils/data_manager.py
    from utils.data_manager import DataManager

except ImportError as e:
    logger.error(f"Failed to import necessary modules. Ensure your project structure matches expectations (config/, adapters/, utils/) and that required files (params.py, paths.py, binance_futures_adapter.py, data_manager.py) exist.")
    logger.error(f"Import Error: {e}")
    sys.exit(1) # Exit if essential imports fail
except FileNotFoundError as e:
    # This specific error might not be caught here if imports are done before setup_rotating_logging
    # but keeping for robustness if import order changes.
    logger.error(f"Configuration file not found: {e}. Ensure config/params.py and config/paths.py exist.")
    sys.exit(1)
except AttributeError as e:
    logger.error(f"Configuration object missing expected attribute or key: {e}. Check config/params.py and config/paths.py.")
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred during initial imports or configuration loading: {e}", exc_info=True)
    sys.exit(1)


# --- Main Fetching Logic ---
# Make the main fetching function asynchronous
async def fetch_and_save_futures_data(symbol: str, interval: str, start_date: str, end_date: Optional[str] = None):
    """
    Fetches historical futures data using BinanceFuturesAdapter and saves it using DataManager.

    Args:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
        interval (str): The time interval for candles (e.g., '1m', '5m', '1h').
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
        end_date (Optional[str]): The end date for fetching data in 'YYYY-MM-DD' format.
                                  Defaults to None (fetches up to current time).
    """
    adapter = None # Initialize adapter to None for finally block
    try:
        # --- Retrieve API Keys from Environment Variables ---
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")

        if not api_key or not api_secret:
            logger.critical("Binance API keys not found in environment variables.")
            logger.critical("Please ensure BINANCE_API_KEY and BINANCE_API_SECRET are set in your environment or in a .env file in the project root.")
            sys.exit(1) # Exit if keys are not set

        # --- Retrieve Adapter Configuration ---
        # Get the specific exchange configuration based on STRATEGY_CONFIG['exchange_type']
        exchange_type = STRATEGY_CONFIG.get('exchange_type', 'binance_futures') # Default to binance_futures
        exchange_adapter_config = EXCHANGE_CONFIG.get(exchange_type, {})

        if not exchange_adapter_config:
            logger.critical(f"Configuration for exchange type '{exchange_type}' not found in EXCHANGE_CONFIG in params.py.")
            sys.exit(1)

        # The BinanceFuturesAdapter.__init__ requires api_key, api_secret, symbol, leverage, logger, and config.
        # For fetch_data, we can use the command-line symbol, a dummy leverage (e.g., 1),
        # and the script's logger. The adapter config comes from EXCHANGE_CONFIG.
        dummy_leverage = 1 # Leverage is not relevant for historical data fetching


        # Initialize the Binance futures adapter with required parameters
        adapter = BinanceFuturesAdapter(
            api_key=api_key,
            api_secret=api_secret,
            symbol=symbol.upper(), # Pass the symbol from command line
            leverage=dummy_leverage, # Pass dummy leverage
            logger=logger, # Pass the script's logger
            config=exchange_adapter_config # Pass the extracted exchange adapter config
        )

        # Perform asynchronous setup for the adapter (e.g., fetching exchange info)
        await adapter.async_setup()

        # Initialize the data manager for saving/loading
        dm = DataManager()

        # Construct the full output path using DataManager's get_file_path
        try:
            output_path = dm.get_file_path(
                symbol=symbol.upper(),
                interval=interval,
                data_type='raw' # Specify the data type
            )
            logger.info(f"Attempting to fetch {symbol.upper()} {interval} futures data from {start_date} to {end_date if end_date else 'current time'} and save to {output_path}...")
        except ValueError as e:
            logger.critical(f"Error constructing output file path: {e}")
            sys.exit(1)


        # Convert start_date and end_date strings to pandas Timestamps (UTC)
        # Ensure timezone awareness for robustness
        try:
            # Localize to UTC if naive, then convert to UTC if not already
            start_ts = pd.to_datetime(start_date).tz_localize('UTC') if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date).tz_convert('UTC')
            end_ts = pd.to_datetime(end_date).tz_localize('UTC') if end_date and pd.to_datetime(end_date).tzinfo is None else (pd.to_datetime(end_date).tz_convert('UTC') if end_date else None)
        except ValueError as e:
            logger.critical(f"Invalid date format provided. Use 'YYYY-MM-DD'. Error: {e}")
            sys.exit(1)


        # --- Await the asynchronous method call ---
        df = await adapter.get_historical_candles(
            symbol=symbol.upper(),
            interval=interval,
            start_time=start_ts,
            end_time=end_ts # Pass the end_time
        )
        # --- End of await ---


        # Check if fetching failed or returned empty data
        if df is None or df.empty:
            logger.warning(f"No data fetched or an error occurred during fetching for {symbol.upper()} {interval} from {start_date} to {end_date if end_date else 'current time'}. No file will be saved.")
            return # Exit if no data was returned

        logger.info(f"Successfully fetched {len(df)} records.")
        logger.info(f"Saving data to {output_path}...")

        # Save the fetched data using the DataManager
        try:
            dm.save_data(
                df_to_save=df,
                symbol=symbol.upper(), # Pass the symbol
                interval=interval,     # Pass the interval
                data_type='raw',       # Specify the data type as 'raw'
            )
            logger.info(f"Successfully saved data for {symbol.upper()} {interval} using DataManager.")
        except Exception as e:
            logger.critical(f"Error saving data for {symbol.upper()} {interval} using DataManager: {e}", exc_info=True)
            sys.exit(1) # Exit on critical save error

    except ConnectionError as ce:
        logger.critical(f"Connection or authentication error with Binance API: {ce}", exc_info=True)
        sys.exit(1)
    except ValueError as ve:
        logger.critical(f"Value error during data fetching or processing: {ve}", exc_info=True)
        sys.exit(1)
    except BinanceAPIException as bae:
        logger.critical(f"Binance API error during data fetching: {bae}", exc_info=True)
        sys.exit(1)
    except BinanceRequestException as bre:
        logger.critical(f"Binance request error during data fetching: {bre}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during data fetching or saving: {e}", exc_info=True)
        sys.exit(1) # Exit on any unexpected error
    finally:
        # Ensure the adapter connection is closed, even if an error occurs
        if adapter:
            await adapter.close_connection()


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fetch historical futures market data from Binance and save it.'
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
        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'], # Expanded choices
        help='Time interval for candles (e.g., 5m, 1h, 1d)'
    )
    parser.add_argument(
        '--start_date',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format (e.g., 2023-01-01)'
    )
    parser.add_argument(
        '--end_date',
        type=str,
        required=False, # Make end_date optional
        help='End date in YYYY-MM-DD format (e.g., 2024-01-01). Defaults to current time.'
    )


    args = parser.parse_args()

    # --- Run the asynchronous fetching function ---
    try:
        asyncio.run(fetch_and_save_futures_data(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        ))
    except Exception as e:
        logger.critical(f"An error occurred during the asyncio run: {e}", exc_info=True)
        sys.exit(1)

    """
    Usage example:
    1. Create a .env file in the project root with BINANCE_API_KEY and BINANCE_API_SECRET.
    2. Ensure config/params.py contains relevant configuration in STRATEGY_CONFIG['exchange_type']
       and EXCHANGE_CONFIG['binance_futures'].
    3. Ensure config/paths.py contains 'raw_data_dir' and 'raw_data_pattern' keys in the PATHS dictionary.
    4. Run the script from the project root:
       python scripts/fetch_data.py --symbol ADAUSDT --interval 5m --start_date 2024-01-01 --end_date 2024-03-01

    To fetch data up to the current time:
       python scripts/fetch_data.py --symbol ADAUSDT --interval 5m --start_date 2024-01-01
    """
