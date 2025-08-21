#!/usr/bin/env python3
"""
Script to fetch historical futures market data from Binance
using the BinanceFuturesAdapter and save it using the DataManager.

This script leverages the new modular configuration structure (AppConfig)
and validates the exchange configuration before fetching.
"""

import argparse
from pathlib import Path
import logging
import sys
import asyncio
from dotenv import load_dotenv
from binance.exceptions import BinanceAPIException, BinanceRequestException
from typing import Optional
import pandas as pd

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration and Imports ---
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the centralized AppConfig and the validator
try:
    from config.params import app_config
    from config.validator import validate_config
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import configuration modules. Error: {e}", file=sys.stderr)
    sys.exit(1)

# Import the logger setup utility
try:
    from utils.logger_config import setup_rotating_logging
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import logger utility. Error: {e}", file=sys.stderr)
    sys.exit(1)

# --- Set up Logging ---
try:
    setup_rotating_logging('fetch_data')
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"WARNING: Failed to configure rotating logging: {e}. Using basic stdout logging.", file=sys.stderr)

logger = logging.getLogger(__name__)

# Import project-specific modules
try:
    from utils.exchange_adapters.binance.futures_adapter import BinanceFuturesAdapter
    from utils.data_management.data_manager import DataManager
except ImportError as e:
    logger.critical(f"Failed to import necessary modules: {e}", exc_info=True)
    sys.exit(1)


# --- Main Fetching Logic ---
async def fetch_and_save_futures_data(symbol: str, interval: str, start_date: str, end_date: Optional[str] = None):
    """
    Fetches historical futures data using BinanceFuturesAdapter and saves it using DataManager.

    Args:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
        interval (str): The time interval for candles (e.g., '1m', '5m', '1h').
        start_date (str): The start date for fetching data in 'YYYY-MM-DD' format.
        end_date (Optional[str]): The end date for fetching data in 'YYYY-MM-DD' format.
    """
    adapter = None
    try:
        # --- Validate Exchange Configuration ---
        logger.info("Validating exchange configuration...")
        validate_config(app_config.exchange)
        logger.info("Exchange configuration is valid.")

        if not app_config.exchange.api_key or not app_config.exchange.api_secret:
            logger.critical("Binance API keys not found. Please set BINANCE_API_KEY and BINANCE_API_SECRET.")
            sys.exit(1)

        # --- UPDATED: Initialize the new BinanceFuturesAdapter ---
        # The adapter now takes the entire app_config object, from which it
        # extracts the API keys and other necessary settings itself.
        adapter = BinanceFuturesAdapter(
            app_config=app_config,
            symbol=symbol.upper(),
            logger=logger
        )

        # The async_setup method now handles connection, fetching exchange info, and setting leverage.
        await adapter.async_setup()

        dm = DataManager()

        logger.info(f"Attempting to fetch {symbol.upper()} {interval} futures data from {start_date} to {end_date or 'current time'}...")

        start_ts = pd.to_datetime(start_date, utc=True)
        end_ts = pd.to_datetime(end_date, utc=True) if end_date else None

        df = await adapter.get_historical_candles(
            symbol=symbol.upper(),
            interval=interval,
            start_time=start_ts,
            end_time=end_ts
        )

        if df is None or df.empty:
            logger.warning(f"No data fetched for {symbol.upper()} {interval}. No file will be saved.")
            return

        logger.info(f"Successfully fetched {len(df)} records.")

        # Use the new, safer save_dataframe method from our previous refactoring
        dm.save_dataframe(
            df=df,
            data_type='raw',
            symbol=symbol.upper(),
            interval=interval
        )

    except (ValueError, TypeError) as e:
        logger.critical(f"Configuration validation failed or invalid date format: {e}", exc_info=True)
        sys.exit(1)
    except (BinanceAPIException, BinanceRequestException) as e:
        logger.critical(f"Binance API or request error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if adapter:
            await adapter.close_connection()
            logger.info("Exchange connection closed.")


# --- Script Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fetch historical futures market data from Binance and save it.'
    )
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument(
        '--interval', type=str, required=True,
        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Time interval for candles (e.g., 5m, 1h, 1d)'
    )
    parser.add_argument('--start_date', type=str, required=True, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, required=False, help='End date in YYYY-MM-DD format. Defaults to current time.')

    args = parser.parse_args()

    try:
        asyncio.run(fetch_and_save_futures_data(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        ))
    except KeyboardInterrupt:
        logger.info("Script interrupted by user.")
        sys.exit(0)