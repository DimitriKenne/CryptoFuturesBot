#!/usr/bin/env python3
"""
monte_carlo_backtest.py

Orchestrates an advanced Monte Carlo backtesting workflow.
This script has been refactored to use the modular, pure-function utilities.

Workflow:
1.  Loads full historical data using a DataManager.
2.  Splits data into 'garch_fitting_data' and 'simulation_baseline_data' based on the chosen mode.
3.  Runs a standard, deterministic backtest on 'simulation_baseline_data' to establish a baseline for comparison.
4.  Initializes the PricePathSimulator with 'garch_fitting_data'.
5.  Loops for a specified number of simulations:
    a. Generates a synthetic OHLCV data path.
    b. Runs the Backtester with the synthetic data.
    c. Uses PerformanceAnalyzer to calculate metrics for that single run.
    d. Stores the metrics and equity curve from each simulation.
6.  Initializes MonteCarloAnalyzer with all collected results.
7.  Calls MonteCarloAnalyzer to generate aggregate tables and plots.
8.  Calls a single method in DataManager to save all artifacts to a unique, timestamped directory.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from tqdm import tqdm

# --- IMPORTANT: Set Matplotlib backend BEFORE importing pyplot ---
import matplotlib
matplotlib.use('Agg')

# --- Add Project Root to sys.path ---
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import Project Modules ---
try:
    from config.params import app_config, AppConfig
    from config.validator import validate_config
    from utils.data_management.data_manager import DataManager
    from utils.strategy_evaluation.performance_analyzer import PerformanceAnalyzer
    from utils.logger_config import setup_rotating_logging
    from utils.strategy_execution.backtester import Backtester
    from utils.strategy_evaluation.price_path_simulator import PricePathSimulator
    from utils.strategy_evaluation.monte_carlo_analyzer import MonteCarloAnalyzer
except ImportError as e:
    print(f"ERROR: Failed to import necessary project modules: {e}", file=sys.stderr)
    print("Please ensure all dependencies are installed and paths are correct.", file=sys.stderr)
    sys.exit(1)



# --- Logger Setup ---
setup_rotating_logging("mc_backtest")
logger = logging.getLogger(__name__)

# --- Validate Config ---
logger.info("Validating application configuration...")
validate_config(app_config)
logger.info("Configuration is valid.")


def run_mc_backtest_pipeline(symbol: str, interval: str, model_type: str, backtest_mode: str, train_ratio: float, num_simulations: int, num_plot_simulations: int):
    """
    Executes the full, refactored Monte Carlo backtesting pipeline.
    """
    logger.info(f"\n--- Starting Monte Carlo Backtest Run ---")
    logger.info(f"Symbol: {symbol}, Interval: {interval}, Model: {model_type}, Mode: {backtest_mode}")
    logger.info(f"Number of Simulations: {num_simulations}, Number to Plot: {num_plot_simulations}")
    
    # --- NEW: Override the number of plot simulations in the config ---
    app_config.trading.backtest.monte_carlo_plot_simulations = num_plot_simulations
    
    data_manager = DataManager()

    # --- 1. Load Full Historical Data ---
    logger.info("📦 Loading full historical RAW data...")
    full_historical_data = data_manager.load_dataframe(
        data_type='raw',
        symbol=symbol,
        interval=interval
    )
    if full_historical_data is None or full_historical_data.empty:
        logger.critical(f"Historical raw data not found or empty for {symbol} {interval}.")
        sys.exit(1)

    if not isinstance(full_historical_data.index, pd.DatetimeIndex):
        full_historical_data.index = pd.to_datetime(full_historical_data.index, utc=True)
    if full_historical_data.index.freq is None:
        full_historical_data.index.freq = pd.infer_freq(full_historical_data.index)
        if full_historical_data.index.freq is None:
            logger.critical("Could not infer data frequency. Cannot proceed with simulation.")
            sys.exit(1)
        logger.info(f"Inferred data frequency: {full_historical_data.index.freq}")

    # --- 2. Determine Data Segments ---
    garch_fitting_data: pd.DataFrame
    simulation_baseline_data: pd.DataFrame

    split_index = int(len(full_historical_data) * train_ratio)
    
    if backtest_mode == 'test':
        garch_fitting_data = full_historical_data.iloc[:split_index]
        simulation_baseline_data = full_historical_data.iloc[split_index:]
        logger.info(f"Mode 'test': GARCH fitted on {len(garch_fitting_data)} train bars. Simulations/Baseline on {len(simulation_baseline_data)} test bars.")
    elif backtest_mode == 'train':
        garch_fitting_data = full_historical_data
        simulation_baseline_data = full_historical_data.iloc[:split_index]
        logger.info(f"Mode 'train': GARCH fitted on {len(garch_fitting_data)} full bars. Simulations/Baseline on {len(simulation_baseline_data)} train bars.")
    else: # 'full' mode
        garch_fitting_data = full_historical_data
        simulation_baseline_data = full_historical_data
        logger.info(f"Mode 'full': GARCH fitted on {len(garch_fitting_data)} full bars. Simulations/Baseline on {len(simulation_baseline_data)} full bars.")

    if simulation_baseline_data.empty or garch_fitting_data.empty:
        logger.critical("Data splitting resulted in an empty dataframe for GARCH fitting or simulation. Exiting.")
        sys.exit(1)

    # --- 3. Run Deterministic Backtest for Baseline ---
    logger.info("--- Running Deterministic Backtest for Baseline ---")
    deterministic_results = {}
    try:
        det_backtester = Backtester(
            app_config=app_config,
            symbol=symbol,
            interval=interval,
            model_type=model_type,
            backtest_mode='full',
            initial_ohlcv_data=simulation_baseline_data.copy()
        )
        det_trades, det_equity = det_backtester.run_backtest()
        
        if det_trades.empty or det_equity.empty:
            logger.critical("Deterministic backtest produced no trades or an empty equity curve. Cannot establish a valid baseline. Aborting.")
            sys.exit(1)

        det_analyzer = PerformanceAnalyzer(
            app_config=app_config,
            trade_history_df=det_trades,
            equity_df=det_equity,
            symbol=symbol,
            interval=interval,
            model_type=model_type
        )
        det_metrics, _ = det_analyzer.generate_analysis_artifacts()
        
        deterministic_results = {
            'metrics': det_metrics,
            'equity_curve': det_equity,
            'ohlcv_data': simulation_baseline_data.copy()
        }
        logger.info("Deterministic backtest complete. Results stored for comparison.")
    except Exception as e:
        logger.critical(f"Deterministic backtest failed: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Initialize PricePathSimulator ---
    simulator = PricePathSimulator(garch_fitting_data, app_config)
    logger.info(f"PricePathSimulator initialized, fitted on {len(garch_fitting_data)} bars.")

    # --- 5. Run Monte Carlo Simulation Loop ---
    all_metrics: List[Dict[str, Any]] = []
    all_equity_curves: List[pd.Series] = []
    all_simulated_paths: List[pd.DataFrame] = []
    logger.info(f"--- Starting {num_simulations} Monte Carlo Simulations ---")
    
    for i in tqdm(range(num_simulations), desc="Running Backtest Simulations"):
        try:
            synthetic_ohlcv_df = simulator.simulate_one_path(
                num_periods=len(simulation_baseline_data),
                start_date=simulation_baseline_data.index[0],
                freq=simulation_baseline_data.index.freq
            )
            if synthetic_ohlcv_df is None:
                logger.warning(f"Simulation {i+1}: Simulator returned None. Skipping.")
                continue
            
            all_simulated_paths.append(synthetic_ohlcv_df.copy())

            sim_backtester = Backtester(
                app_config=app_config,
                symbol=symbol,
                interval=interval,
                model_type=model_type,
                backtest_mode='full',
                initial_ohlcv_data=synthetic_ohlcv_df
            )
            sim_trades, sim_equity = sim_backtester.run_backtest()

            sim_equity_series = sim_equity['equity'] if isinstance(sim_equity, pd.DataFrame) and 'equity' in sim_equity else pd.Series(dtype=float, name='equity')

            sim_analyzer = PerformanceAnalyzer(
                app_config=app_config,
                trade_history_df=sim_trades,
                equity_df=sim_equity,
                symbol=symbol,
                interval=interval,
                model_type=model_type
            )
            sim_metrics, _ = sim_analyzer.generate_analysis_artifacts()
            
            all_metrics.append(sim_metrics)
            all_equity_curves.append(sim_equity_series)

        except Exception as e:
            logger.error(f"Backtest on simulation {i+1} failed: {e}", exc_info=False)
            all_metrics.append({'error': str(e)})
            all_equity_curves.append(pd.Series(dtype=float, name='equity'))
            all_simulated_paths.append(pd.DataFrame())

    # --- 6. Aggregate and Analyze Results ---
    if not all_metrics:
        logger.error("No simulations were successfully completed. Cannot perform aggregate analysis. Exiting.")
        return

    logger.info("Aggregating results with MonteCarloAnalyzer...")
    mc_analyzer = MonteCarloAnalyzer(
        app_config=app_config,
        metrics_df=pd.DataFrame(all_metrics),
        all_equity_curves=all_equity_curves,
        deterministic_results=deterministic_results,
        all_simulated_paths=all_simulated_paths
    )
    summary_tables, plots_dict = mc_analyzer.generate_analysis_artifacts()
    logger.info("Analysis complete. Summary tables and plot figures generated.")

    # --- 7. Save All Artifacts ---
    logger.info("Saving all Monte Carlo artifacts...")
    data_manager.save_monte_carlo_artifacts(
        model_type=model_type,
        symbol=symbol,
        interval=interval,
        mode=backtest_mode,
        num_simulations=num_simulations,
        summary_tables=summary_tables,
        plots_dict=plots_dict,
        app_config=app_config
    )
    logger.info("All artifacts saved successfully to the Monte Carlo run directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run advanced Monte Carlo backtests on a trained model.")
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, required=True, help='Time interval (e.g., 1h, 1d)')
    parser.add_argument('--model', type=str, required=True, choices=list(app_config.model.AVAILABLE_MODEL_TYPES.keys()), help='Model key from app_config.model')
    parser.add_argument('--backtest_mode', type=str, default='test', choices=['full', 'train', 'test'], help='Data split to use for GARCH fitting and simulation length.')
    parser.add_argument('--train_ratio', type=float, default=app_config.model.train_test_split_ratio, help='Train/test split ratio.')
    parser.add_argument('--num_simulations', type=int, default=100, help='Number of Monte Carlo simulations to run.')
    # --- NEW: Argument to control the number of plotted simulations ---
    parser.add_argument('--num_plot_simulations', type=int, default=app_config.trading.backtest.monte_carlo_plot_simulations, 
                        help='Number of simulation paths to display on plots.')
    
    args = parser.parse_args()

    try:
        run_mc_backtest_pipeline(
            symbol=args.symbol.upper(),
            interval=args.interval,
            model_type=args.model,
            backtest_mode=args.backtest_mode,
            train_ratio=args.train_ratio,
            num_simulations=args.num_simulations,
            num_plot_simulations=args.num_plot_simulations # Pass the new argument
        )
    except Exception as e:
        logger.critical(f"Unhandled exception in pipeline: {e}", exc_info=True)
    finally:
        logging.shutdown()
        logger.info("\n--- Monte Carlo Backtest Script Finished ---")