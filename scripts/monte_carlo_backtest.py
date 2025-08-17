#!/usr/bin/env python3
"""
monte_carlo_backtest.py

Orchestrates an advanced Monte Carlo backtesting workflow.

Workflow:
1.  Loads historical data and splits it into train/test sets based on user-defined mode.
2.  Runs a standard, deterministic backtest on the actual test data to establish a baseline.
3.  Fits a GARCH(1,1) model to the returns of the training data set (diffusion component).
4.  Estimates jump parameters from GARCH residuals.
5.  Generates synthetic OHLCV data paths incorporating GARCH-modeled diffusion and a Poisson-driven jump process.
6.  Loops for a specified number of simulations:
    a. Generates a synthetic OHLCV data path using PricePathSimulator.
    b. **Utilizes MarketDataHandler (via Backtester) to process this synthetic data, generate features and signals.**
    c. Runs a backtest using the modular Backtester with the processed synthetic data.
    d. Stores the summary metrics and the full equity curve from each run.
7.  Aggregates all results and uses MonteCarloAnalyzer to generate and save:
    a. Statistical summary of performance metrics (CSV).
    b. Distribution plots for key metrics (e.g., Total Return).
    c. A comparative plot of simulated equity curves vs. the deterministic baseline.
    d. A risk/reward scatter plot (Return vs. Drawdown).
    e. A sample of simulated OHLCV paths.
8.  Saves all artifacts to a dedicated, non-conflicting subdirectory.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm # For progress bars

# --- IMPORTANT: Set Matplotlib backend BEFORE importing pyplot ---
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns # For enhanced plotting

# --- Add Project Root to sys.path ---
# This ensures that imports like 'config.params' and 'utils.<module>' work correctly
try:
    script_dir = Path(__file__).resolve().parent
    PROJECT_ROOT = script_dir.parent
except NameError:
    # Fallback for environments where __file__ might not be defined (e.g., some interactive shells)
    PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import Project Modules (Updated for new, separated utilities) ---
try:
    from config.params import app_config, AppConfig # Import AppConfig for type hinting
    from config.paths import PATHS
    from utils.data_management.market_data_handler import MarketDataHandler # The data processing pipeline
    from utils.analysis.performance_analyzer import PerformanceAnalyzer # General purpose analyzer for deterministic run
    from utils.logger_config import setup_rotating_logging # Centralized logging setup
    from utils.strategy_execution.backtester import Backtester # The modular Backtester
    from utils.simulation.price_path_simulator import PricePathSimulator # New: Imported utility
    from utils.analysis.monte_carlo_analyzer import MonteCarloAnalyzer # New: Imported utility
except ImportError as e:
    print(f"ERROR: Failed to import necessary project modules: {e}", file=sys.stderr)
    print("Please ensure all dependencies are installed (including 'tqdm', 'seaborn', 'arch') and paths are correct.", file=sys.stderr)
    sys.exit(1)

# --- Logger Setup ---
setup_rotating_logging("mc_backtest")
logger = logging.getLogger(__name__)


def run_mc_backtest_pipeline(symbol: str, interval: str, model_key: str, backtest_mode: str, train_ratio: float, num_simulations: int):
    """
    Executes the full Monte Carlo backtesting pipeline.

    Args:
        symbol (str): The trading pair symbol (e.g., "BTCUSDT").
        interval (str): The data interval (e.g., "1h").
        model_key (str): The key for the ML model type to use (e.g., "xgboost").
        backtest_mode (str): Data split mode ('full', 'train', 'test') for historical data.
        train_ratio (float): Ratio for train-test split.
        num_simulations (int): Number of Monte Carlo simulations to run.
    """
    logger.info(f"\n--- Starting Monte Carlo Backtest Run ---")
    logger.info(f"Symbol: {symbol}, Interval: {interval}, Model: {model_key}, Mode: {backtest_mode}")
    logger.info(f"Number of Simulations: {num_simulations}")

    # --- 1. Load and Split Historical Data ---
    # Use MarketDataHandler to load the full historical processed data.
    # We create a temporary instance just for loading and splitting the data for GARCH fitting.
    market_data_loader_instance = MarketDataHandler(app_config=app_config)
    try:
        # Use the internal _load_data_for_processing method to get the full historical dataset.
        full_historical_data = market_data_loader_instance._load_data_for_processing(
            symbol=symbol,
            interval=interval,
            model_type=model_key, # Pass model_type to ensure correct path resolution if needed
            backtest_mode='full', # Always load full data for initial splitting
            train_ratio=1.0 # Not relevant when loading full
        )

        if full_historical_data is None or full_historical_data.empty:
            raise FileNotFoundError(f"Historical processed data not found or empty for {symbol} {interval}.")
        
        # Ensure data index is DatetimeIndex and has a frequency. This is crucial for pd.date_range.
        if not isinstance(full_historical_data.index, pd.DatetimeIndex):
            full_historical_data.index = pd.to_datetime(full_historical_data.index, utc=True)
        if full_historical_data.index.freq is None:
            inferred_freq = pd.infer_freq(full_historical_data.index)
            if inferred_freq:
                full_historical_data.index.freq = inferred_freq
                logger.info(f"Inferred data frequency: {inferred_freq}")
            else:
                logger.warning("Could not infer data frequency. Using 'min' as a fallback. This might cause issues with date_range generation.")
                full_historical_data.index.freq = 'min' # Fallback to minute frequency

        # Split data into training and testing sets based on the requested mode.
        if backtest_mode == 'full':
            train_data, test_data = full_historical_data, full_historical_data
        else:
            train_size = int(len(full_historical_data) * train_ratio)
            train_data = full_historical_data.iloc[:train_size]
            test_data = full_historical_data.iloc[train_size:]
        
        num_periods_to_simulate = len(test_data)
        logger.info(f"Data split: {len(train_data)} train bars, {len(test_data)} test bars (for simulation length).")
        if num_periods_to_simulate == 0:
            raise ValueError("Test data set is empty after splitting. Adjust train_ratio or ensure sufficient data.")

    except Exception as e:
        logger.critical(f"Data loading/splitting failed: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Run Deterministic Backtest for Baseline ---
    logger.info("--- Running Deterministic Backtest for Baseline ---")
    deterministic_results = {}
    try:
        # Instantiate Backtester for the deterministic run, which uses MarketDataHandler internally
        det_backtester = Backtester(
            app_config=app_config,
            symbol=symbol,
            interval=interval,
            model_type=model_key,
            backtest_mode=backtest_mode, # Use the selected backtest_mode for deterministic run
            train_ratio=train_ratio
        )
        
        # Run the backtest and capture the results (trade history, equity curve, summary metrics)
        det_trades, det_equity, det_metrics = det_backtester.run_backtest()
        
        # Store relevant parts of the deterministic results for the Monte Carlo Analyzer
        deterministic_results = {
            'trades': det_trades,
            'equity_curve': det_equity,
            'metrics': det_metrics,
            'ohlcv_data': test_data.copy(), # Store the actual OHLCV test data for comparison plots
            'config_symbol': symbol,
            'config_interval': interval
        }
        logger.info("Deterministic backtest complete. Results stored for comparison.")
    except Exception as e:
        logger.critical(f"Deterministic backtest failed, cannot proceed with Monte Carlo: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Initialize PricePathSimulator (GARCH + Jumps) ---
    if train_data.empty or len(train_data) < 2: # Need at least 2 bars for pct_change for GARCH fitting
        logger.critical("Training data is too short or empty for PricePathSimulator to fit GARCH model. Exiting.")
        sys.exit(1)

    simulator = PricePathSimulator(train_data, app_config) # Pass train_data to fit the GARCH model
    logger.info(f"PricePathSimulator initialized for GARCH + Jumps simulation.")


    # --- 4. Run Monte Carlo Simulation Loop ---
    all_metrics: List[Dict[str, Any]] = []
    all_equity_curves: List[pd.Series] = []
    all_simulated_paths: List[pd.DataFrame] = [] # To store the raw synthetic OHLCV paths
    logger.info(f"--- Starting {num_simulations} Monte Carlo Simulations ---")
    
    for i in tqdm(range(num_simulations), desc="Running Backtest Simulations"):
        sim_summary_metrics: Dict[str, Any] = {}
        sim_equity_curve = pd.Series(dtype=float) # Initialize as empty Series

        try:
            # Generate one synthetic raw OHLCV data path for this simulation
            # The length of this path matches the test_data length
            synthetic_ohlcv_df = simulator.simulate_one_path(
                num_periods=num_periods_to_simulate,
                start_date=test_data.index[0], # Start date matches the real test data
                freq=test_data.index.freq # Frequency matches the real test data
            )

            if synthetic_ohlcv_df is None or synthetic_ohlcv_df.empty:
                logger.warning(f"Simulation {i+1}: Generated empty synthetic OHLCV data. Skipping this simulation.")
                sim_summary_metrics['error'] = 'Empty synthetic OHLCV data'
                all_metrics.append(sim_summary_metrics)
                all_equity_curves.append(sim_equity_curve)
                all_simulated_paths.append(pd.DataFrame()) # Append an empty path on failure
                continue
            
            all_simulated_paths.append(synthetic_ohlcv_df.copy()) # Store the generated raw path

            # Initialize a fresh Backtester instance for each simulation.
            # Crucially, pass the `synthetic_ohlcv_df` to the Backtester's MarketDataHandler
            # via the `initial_ohlcv_data` parameter in `get_processed_data_stream`.
            # This ensures the Backtester processes our in-memory synthetic data.
            sim_backtester = Backtester(
                app_config=app_config,
                symbol=symbol,
                interval=interval,
                model_type=model_key,
                backtest_mode='full', # Treat synthetic data as a 'full' dataset for this backtest
                train_ratio=1.0 # Not used when initial_ohlcv_data is provided
            )

            # Override the Backtester's internal MarketDataHandler's data stream source
            # to use our synthetic data instead of loading from files.
            # This is the "clean" way to inject in-memory data into the Backtester's pipeline.
            sim_backtester.market_data_handler.get_processed_data_stream = \
                lambda sym, intr, m_type, b_mode, t_ratio: \
                    MarketDataHandler(app_config).get_processed_data_stream(
                        symbol=sym, interval=intr, model_type=m_type, backtest_mode=b_mode,
                        train_ratio=t_ratio, initial_ohlcv_data=synthetic_ohlcv_df # Pass the synthetic data here
                    )

            # Run the backtest on the synthetic data and capture its results
            sim_trades, sim_equity_curve, temp_summary_metrics = sim_backtester.run_backtest()
            
            # Update the metrics list with results from this simulation
            sim_summary_metrics.update(temp_summary_metrics)
            all_metrics.append(sim_summary_metrics)
            all_equity_curves.append(sim_equity_curve)

        except Exception as e:
            logger.error(f"Backtest on simulation {i+1} failed: {e}", exc_info=False) # Log without full traceback unless debug needed
            sim_summary_metrics['error'] = str(e) # Record the error in metrics
            all_metrics.append(sim_summary_metrics)
            all_equity_curves.append(sim_equity_curve) # Append current (potentially empty/partial) equity curve
            all_simulated_paths.append(pd.DataFrame()) # Append empty path if error occurs


    # --- 5. Aggregate and Analyze Results ---
    if not all_metrics:
        logger.error("No simulations were successfully completed. Cannot perform aggregate analysis. Exiting.")
        return

    # Define a safe and organized output directory for Monte Carlo analysis results
    base_analysis_dir = Path(PATHS.get("backtesting_analysis_dir"))
    output_dir = base_analysis_dir / model_key / f"{symbol.replace('/', '_')}_{interval}_monte_carlo_garch_jumps"
    
    # Instantiate MonteCarloAnalyzer and run its full analysis pipeline
    analyzer = MonteCarloAnalyzer(pd.DataFrame(all_metrics), all_equity_curves, deterministic_results, output_dir, all_simulated_paths)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Run advanced Monte Carlo backtests on a trained model.")
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, required=True, choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'], help='Time interval (e.g., 1h, 1d)')
    parser.add_argument('--model', type=str, required=True, choices=list(app_config.model.AVAILABLE_MODEL_TYPES.keys()), help='Model key from app_config.model')
    parser.add_argument('--backtest_mode', type=str, default='test', choices=['full', 'train', 'test'], help='Data split to use for GARCH fitting and simulation length.')
    parser.add_argument('--train_ratio', type=float, default=app_config.general.train_test_split_ratio, help='Train/test split ratio.')
    parser.add_argument('--num_simulations', type=int, default=100, help='Number of Monte Carlo simulations to run.')
    
    args = parser.parse_args()

    try:
        run_mc_backtest_pipeline(
            symbol=args.symbol.upper(),
            interval=args.interval,
            model_key=args.model,
            backtest_mode=args.backtest_mode,
            train_ratio=args.train_ratio,
            num_simulations=args.num_simulations
        )
    except Exception as e:
        logger.critical(f"Unhandled exception in pipeline: {e}", exc_info=True)
    finally:
        # Ensure all log handlers are properly closed on script exit
        logging.shutdown()
