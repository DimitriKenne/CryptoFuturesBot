#!/usr/bin/env python3
# scripts/backtest.py

"""
Main script to run a backtest simulation for the trading bot.
Parses command-line arguments and orchestrates the Backtester class.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the main AppConfig
from config.params import app_config

# Import the Backtester class and PerformanceAnalyzer
from utils.strategy_execution.backtester import Backtester
from utils.analysis.performance_analyzer import PerformanceAnalyzer # Import PerformanceAnalyzer

# Import the logging setup function
from utils.logger_config import setup_rotating_logging

# --- Logging Setup ---
# Setup for basic console logging. For more advanced logging,
# integrate a dedicated logging utility.
setup_rotating_logging("backtest") # Pass "backtest" as the positional argument
logger = logging.getLogger(__name__) # Re-get logger to use the configured handlers


def main():
    """
    Main function to parse arguments and run the backtest.
    """
    parser = argparse.ArgumentParser(description="Run a backtest simulation for the trading bot.")
    parser.add_argument('--symbol', type=str, required=True,
                        help='Trading symbol (e.g., BTCUSDT).')
    parser.add_argument('--interval', type=str, required=True,
                        help='Data interval (e.g., 1h, 5m).')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(app_config.model.AVAILABLE_MODEL_TYPES.keys()),
                        help=f"Model type to use for signals. Available: {list(app_config.model.AVAILABLE_MODEL_TYPES.keys())}")
    parser.add_argument('--backtest_mode', type=str, default='test',
                        choices=['full', 'train', 'test'],
                        help='Mode for backtest data split: full dataset, training set, or test set.')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of data to use for training split (0.0 to 1.0). Only applicable in "train" or "test" backtest_mode.')

    args = parser.parse_args()

    logger.info(f"--- Backtest Script Started ({args.symbol} {args.interval} {args.model}) ---")

    try:
        backtester = Backtester(
            app_config=app_config,
            symbol=args.symbol,
            interval=args.interval,
            model_type=args.model,
            backtest_mode=args.backtest_mode,
            train_ratio=args.train_ratio
        )
        # Run backtest and capture the results (returns PerformanceAnalyzer now)
        final_trade_history, final_equity_curve, performance_analyzer = backtester.run_backtest()

        # We can save the results (trade history and equity curve)
        backtester.save_results()

        # Only run full analysis (metrics + plots) for deterministic backtest
        logger.info("Running Performance Analysis for the deterministic backtest...")
        performance_analyzer.run_full_analysis() # This will calculate, save, and plot

    except SystemExit:
        logger.info("Backtest script finished as requested.")
    except Exception as e:
        logger.critical(f"Unhandled exception during backtest execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.shutdown()
        logger.info("--- Backtest Script Finished ---")

if __name__ == "__main__":
    main()

