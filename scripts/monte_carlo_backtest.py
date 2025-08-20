#!/usr/bin/env python3
"""
monte_carlo_backtest.py

Orchestrates an advanced Monte Carlo backtesting workflow.

Workflow:
1.  Loads full historical data.
2.  Splits data into relevant segments based on 'backtest_mode' and 'train_ratio'.
    - If backtest_mode='test': Splits into train_data (for GARCH fitting) and test_data (for simulation length/baseline).
    - If backtest_mode='train' or 'full': The 'full' historical data is used for GARCH fitting, and the simulation
      length/baseline is determined by the 'train_data' (if 'train' mode) or 'full_historical_data' (if 'full' mode).
3.  Runs a standard, deterministic backtest on the relevant baseline data to establish a comparison.
4.  Fits a GARCH(1,1)+jump model to the **appropriate historical data segment** (train_data or full_historical_data).
5.  Generates synthetic OHLCV data paths incorporating the fitted GARCH-modeled diffusion and a Poisson-driven jump process.
6.  Loops for a specified number of simulations:
    a. Generates a synthetic OHLCV data path using PricePathSimulator, matching the length of the chosen baseline data.
    b. Utilizes the modular Backtester directly with this synthetic data for processing and signal generation.
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

    # --- 1. Load Full Historical Data for Splitting and GARCH Fitting ---
    logger.info("📦 Loading full historical data for GARCH fitting and simulation baseline...")
    # Use MarketDataHandler to load the full historical processed data.
    # We set backtest_mode to 'full' here to ensure the MarketDataHandler doesn't split it internally
    # when fetching data for the simulator.
    temp_mdh = MarketDataHandler(
        app_config=app_config,
        mode='backtest',
        symbol=symbol,
        interval=interval,
        model_type=model_key,
        train_ratio=1.0, # Not relevant here, but for MDH init
        backtest_mode='full' # Ensures full raw data is loaded from disk
    )
    try:
        # Use the internal _load_data_for_processing method to get the full historical dataset.
        full_historical_data = temp_mdh._load_data_for_processing()

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

    except Exception as e:
        logger.critical(f"Data loading failed for Monte Carlo pipeline: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Determine Data Segments for GARCH Fitting and Simulation Baseline ---
    garch_fitting_data: pd.DataFrame
    simulation_baseline_data: pd.DataFrame # This defines length and start date for simulations and deterministic baseline

    if backtest_mode == 'test':
        train_size = int(len(full_historical_data) * train_ratio)
        if train_size == 0:
            logger.critical("Train size is 0 for 'test' mode. Cannot fit GARCH model. Adjust train_ratio or data.")
            sys.exit(1)
        if len(full_historical_data) - train_size == 0:
             logger.critical("Test size is 0 for 'test' mode. Cannot run simulations. Adjust train_ratio or data.")
             sys.exit(1)

        train_data = full_historical_data.iloc[:train_size]
        test_data = full_historical_data.iloc[train_size:]

        garch_fitting_data = train_data # GARCH fitted on train_data for realistic backtesting
        simulation_baseline_data = test_data # Simulations and deterministic baseline run on test_data
        logger.info(f"Mode 'test': GARCH fitted on {len(garch_fitting_data)} train bars. Simulations/Baseline on {len(simulation_baseline_data)} test bars.")

    elif backtest_mode == 'train':
        train_size = int(len(full_historical_data) * train_ratio)
        if train_size == 0:
            logger.critical("Train size is 0 for 'train' mode. Cannot run simulations or baseline. Adjust train_ratio or data.")
            sys.exit(1)

        train_data = full_historical_data.iloc[:train_size]
        
        garch_fitting_data = full_historical_data # GARCH fitted on full_historical_data for stress testing
        simulation_baseline_data = train_data # Simulations and deterministic baseline run on train_data
        logger.info(f"Mode 'train': GARCH fitted on {len(garch_fitting_data)} full bars. Simulations/Baseline on {len(simulation_baseline_data)} train bars.")

    elif backtest_mode == 'full':
        garch_fitting_data = full_historical_data # GARCH fitted on full_historical_data for stress testing
        simulation_baseline_data = full_historical_data # Simulations and deterministic baseline run on full_historical_data
        logger.info(f"Mode 'full': GARCH fitted on {len(garch_fitting_data)} full bars. Simulations/Baseline on {len(simulation_baseline_data)} full bars.")
    else:
        logger.critical(f"Invalid backtest_mode: {backtest_mode}. Exiting.")
        sys.exit(1)

    if simulation_baseline_data.empty:
        logger.critical("Simulation baseline data is empty. Cannot run Monte Carlo simulations or deterministic backtest. Exiting.")
        sys.exit(1)

    # --- 3. Run Deterministic Backtest for Baseline ---
    logger.info("--- Running Deterministic Backtest for Baseline ---")
    deterministic_results = {}
    try:
        # Pass the specific 'simulation_baseline_data' to the Backtester
        det_backtester = Backtester(
            app_config=app_config,
            symbol=symbol,
            interval=interval,
            model_type=model_key,
            backtest_mode='full', # Treat this specific data as 'full' for the Backtester
            initial_ohlcv_data=simulation_baseline_data # Inject the baseline data
        )
        # Run the backtest and capture the results (returns PerformanceAnalyzer now)
        det_trades, det_equity, det_performance_analyzer = det_backtester.run_backtest()
        det_metrics = det_performance_analyzer.calculate_summary_metrics()
        
        deterministic_results = {
            'trades': det_trades,
            'equity_curve': det_equity,
            'metrics': det_metrics,
            'ohlcv_data': simulation_baseline_data.copy(), # Store the actual OHLCV used for baseline
            'config_symbol': symbol,
            'config_interval': interval,
            'initial_capital': app_config.trading.risk.initial_capital # Explicitly pass initial capital
        }
        logger.info("Deterministic backtest complete. Results stored for comparison.")
    except Exception as e:
        logger.critical(f"Deterministic backtest failed, cannot proceed with Monte Carlo: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Initialize PricePathSimulator (GARCH + Jumps) ---
    if garch_fitting_data.empty or len(garch_fitting_data) < 2: # Need at least 2 bars for pct_change for GARCH fitting
        logger.critical("GARCH fitting data is too short or empty for PricePathSimulator. Exiting.")
        sys.exit(1)

    simulator = PricePathSimulator(garch_fitting_data, app_config) # Pass the determined GARCH fitting data
    logger.info(f"PricePathSimulator initialized for GARCH + Jumps simulation, fitted on {len(garch_fitting_data)} bars.")


    # --- 5. Run Monte Carlo Simulation Loop ---
    all_metrics: List[Dict[str, Any]] = []
    all_equity_curves: List[pd.Series] = []
    all_simulated_paths: List[pd.DataFrame] = [] # To store the raw synthetic OHLCV paths
    logger.info(f"--- Starting {num_simulations} Monte Carlo Simulations ---")
    
    for i in tqdm(range(num_simulations), desc="Running Backtest Simulations"):
        sim_summary_metrics: Dict[str, Any] = {}
        sim_equity_curve = pd.Series(dtype=float) # Initialize as empty Series

        try:
            # Generate one synthetic raw OHLCV data path for this simulation
            # The length of this path matches the determined simulation_baseline_data length
            synthetic_ohlcv_df = simulator.simulate_one_path(
                num_periods=len(simulation_baseline_data),
                start_date=simulation_baseline_data.index[0], # Start date matches the baseline data
                freq=simulation_baseline_data.index.freq # Frequency matches the baseline data
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
            # Crucially, pass the `synthetic_ohlcv_df` directly to the Backtester's `initial_ohlcv_data` parameter.
            # This ensures the Backtester processes our in-memory synthetic data.
            sim_backtester = Backtester(
                app_config=app_config,
                symbol=symbol,
                interval=interval,
                model_type=model_key,
                backtest_mode='full', # Treat synthetic data as a 'full' dataset for the Backtester itself
                initial_ohlcv_data=synthetic_ohlcv_df # Pass the synthetic data here!
            )
            # The MarketDataHandler inside sim_backtester will now use this initial_ohlcv_data.
            # No need to manually override get_processed_data_stream anymore.

            # Run the backtest on the synthetic data and capture its results
            sim_trades, sim_equity_curve, sim_performance_analyzer = sim_backtester.run_backtest()
            sim_metrics = sim_performance_analyzer.calculate_summary_metrics()
            sim_summary_metrics.update(sim_metrics)
            all_metrics.append(sim_summary_metrics)
            all_equity_curves.append(sim_equity_curve)

        except Exception as e:
            logger.error(f"Backtest on simulation {i+1} failed: {e}", exc_info=False) # Log without full traceback unless debug needed
            sim_summary_metrics['error'] = str(e) # Record the error in metrics
            all_metrics.append(sim_summary_metrics)
            all_equity_curves.append(sim_equity_curve) # Append current (potentially empty/partial) equity curve
            all_simulated_paths.append(pd.DataFrame()) # Append empty path if error occurs


    # --- 6. Aggregate and Analyze Results ---
    if not all_metrics:
        logger.error("No simulations were successfully completed. Cannot perform aggregate analysis. Exiting.")
        return

    # Define a safe and organized output directory for Monte Carlo analysis results
    # Ensure this path is distinct from standard backtest analysis results
    base_analysis_dir = Path(PATHS.get("analysis_dir"))
    output_dir = base_analysis_dir / "monte_carlo" / model_key / f"{symbol.replace('/', '_')}_{interval}"
    output_dir = output_dir / f"mode_{backtest_mode}_ratio_{str(train_ratio).replace('.', '')}_sims_{num_simulations}"
    
    # Instantiate MonteCarloAnalyzer and run its full analysis pipeline
    analyzer = MonteCarloAnalyzer(
        metrics_df=pd.DataFrame(all_metrics),
        all_equity_curves=all_equity_curves,
        deterministic_results=deterministic_results,
        output_dir=output_dir, # Pass the specific output directory
        all_simulated_paths=all_simulated_paths,
        app_config=app_config # Pass app_config to the analyzer
    )
    analyzer.run_full_analysis()


if __name__ == "__main__":
    # Argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Run advanced Monte Carlo backtests on a trained model.")
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, required=True, choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'], help='Time interval (e.g., 1h, 1d)')
    parser.add_argument('--model', type=str, required=True, choices=list(app_config.model.AVAILABLE_MODEL_TYPES.keys()), help='Model key from app_config.model')
    parser.add_argument('--backtest_mode', type=str, default='test', choices=['full', 'train', 'test'], help='Data split to use for GARCH fitting and simulation length.')
    parser.add_argument('--train_ratio', type=float, default=app_config.model.train_test_split_ratio, help='Train/test split ratio.')
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
