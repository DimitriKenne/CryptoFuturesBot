#!/usr/bin/env python3
# scripts/backtest.py

"""
Main script to run a backtest simulation for the trading bot.
This script acts as the orchestrator for the entire deterministic backtest process.
It validates config, coordinates the simulation and analysis, and saves all results.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to Python path for imports
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# Import the main AppConfig and refactored utilities
from config.params import app_config
from config.validator import validate_config
from utils.strategy_execution.backtester import Backtester
from utils.strategy_evaluation.performance_analyzer import PerformanceAnalyzer
from utils.data_management.data_manager import DataManager
from utils.logger_config import setup_rotating_logging

# --- Logging Setup ---
setup_rotating_logging("backtest")
logger = logging.getLogger(__name__)


def main():
    """
    Main function to parse arguments and orchestrate the backtest.
    """
    parser = argparse.ArgumentParser(description="Run a backtest simulation for the trading bot.")
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., BTCUSDT).')
    parser.add_argument('--interval', type=str, required=True, help='Data interval (e.g., 1h, 5m).')
    parser.add_argument('--model_type', type=str, required=True, choices=list(app_config.model.AVAILABLE_MODEL_TYPES.keys()),
                        help=f"Model type to use for signals. Available: {list(app_config.model.AVAILABLE_MODEL_TYPES.keys())}")
    parser.add_argument('--backtest_mode', type=str, default='test', choices=['full', 'train', 'test'],
                        help='Mode for backtest data split: full dataset, training set, or test set.')
    parser.add_argument('--train_ratio', type=float, default=app_config.model.train_test_split_ratio,
                        help='Ratio of data to use for training split. Only used in "train" or "test" mode.')

    args = parser.parse_args()

    logger.info(f"--- Backtest Script Started ({args.symbol} {args.interval} {args.model_type} mode: {args.backtest_mode}) ---")

    try:
        # --- 1. Validate Config ---
        logger.info("Validating application configuration...")
        validate_config(app_config)
        logger.info("Configuration is valid.")

        # --- 2. Initialize ---
        data_manager = DataManager()

        # --- 3. Simulate ---
        logger.info("Initializing and running the simulation engine...")
        backtester = Backtester(
            app_config=app_config,
            symbol=args.symbol,
            interval=args.interval,
            model_type=args.model_type,
            backtest_mode=args.backtest_mode,
            train_ratio=args.train_ratio
        )
        trades_df, equity_df = backtester.run_backtest()
        logger.info(f"Simulation complete. Total trades: {len(trades_df)}")

        # --- 4. Analyze ---
        logger.info("Initializing and running the performance analyzer...")
        analyzer = PerformanceAnalyzer(
            app_config=app_config,
            trade_history_df=trades_df,
            equity_df=equity_df,
            symbol=args.symbol,
            interval=args.interval,
            model_type=args.model_type
        )
        metrics_dict, plots_dict = analyzer.generate_analysis_artifacts()
        logger.info("Analysis complete. Metrics and plot figures generated.")
        logger.info(f"Key Metrics: Sharpe Ratio = {metrics_dict.get('Sharpe Ratio', 'N/A'):.4f}, "
                    f"Total Return = {metrics_dict.get('Total Return (%)', 'N/A'):.2f}%, "
                    f"Max Drawdown = {metrics_dict.get('Max Drawdown (%)', 'N/A'):.2f}%")

        # --- 5. Persist ---
        logger.info("Saving all backtest artifacts...")
        data_manager.save_backtest_artifacts(
            model_type=args.model_type,
            symbol=args.symbol,
            interval=args.interval,
            trades_df=trades_df,
            equity_df=equity_df,
            metrics_dict=metrics_dict,
            plots_dict=plots_dict,
            app_config=app_config
        )
        logger.info("All artifacts saved successfully to the backtest run directory.")

    except SystemExit as e:
        logger.info(f"Backtest script finished with exit code: {e.code}")
    except Exception as e:
        logger.critical(f"Unhandled exception during backtest execution: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logging.shutdown()
        logger.info("--- Backtest Script Finished ---")

if __name__ == "__main__":
    main()