#!/usr/bin/env python3
"""
monte_carlo_backtest.py

Orchestrates an advanced Monte Carlo backtesting workflow, designed to feel
like an extension of the deterministic backtester.

Workflow:
1.  Loads historical data and splits it into train/test sets based on user-defined mode.
2.  Runs a standard, deterministic backtest on the actual test data to establish a baseline.
3.  Fits a GARCH(1,1) model to the returns of the training data set.
4.  Loads a pre-trained ML model for signal generation.
5.  Loops for a specified number of simulations:
    a. Generates a synthetic OHLCV data path with the and length as the test set.
    b. Runs the full feature engineering -> signal generation -> backtesting pipeline.
    c. Stores the summary metrics and the full equity curve from each run.
6.  Aggregates all results and uses a dedicated analyzer to generate and save:
    a. Statistical summary of performance metrics (CSV).
    b. Distribution plots for key metrics (e.g., Total Return).
    c. A comparative plot of simulated equity curves vs. the deterministic baseline.
    d. A risk/reward scatter plot (Return vs. Drawdown).
7.  Saves all artifacts to a dedicated, non-conflicting subdirectory to keep results safe.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm

# --- IMPORTANT: Set Matplotlib backend BEFORE importing pyplot ---
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns

# --- Add Project Root to sys.path ---
try:
    script_dir = Path(__file__).resolve().parent
    PROJECT_ROOT = script_dir.parent
except NameError:
    PROJECT_ROOT = Path('.').resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# --- Import Project Modules ---
try:
    from config.params import MODEL_CONFIG, STRATEGY_CONFIG, BACKTESTER_CONFIG, GENERAL_CONFIG, FEATURE_CONFIG
    from config.paths import PATHS
    from utils.data_manager import DataManager
    from utils.model_trainer import ModelTrainer
    from utils.features_engineer import FeaturesEngineer # Corrected class name
    from utils.backtester import Backtester
    from utils.logger_config import setup_rotating_logging
    from arch import arch_model
except ImportError as e:
    print(f"ERROR: Failed to import necessary project modules: {e}", file=sys.stderr)
    print("Please ensure all dependencies are installed (including 'arch', 'tqdm', 'seaborn') and paths are correct.", file=sys.stderr)
    sys.exit(1)

# --- Logger Setup ---
setup_rotating_logging("mc_backtest")
logger = logging.getLogger(__name__)


class GarchOHLCVSimulator:
    """Fits a GARCH model and generates synthetic OHLCV data paths."""
    def __init__(self, historical_data: pd.DataFrame):
        if not all(col in historical_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("Historical data must contain OHLCV columns.")
        self.hist_data = historical_data
        self.hist_returns = historical_data['close'].pct_change().dropna()
        self.fitted_model = None
        self._learn_historical_patterns()

    def _learn_historical_patterns(self):
        """Analyzes historical data to learn distributions for OHLC and Volume."""
        self.hist_hl_range_ratio = ((self.hist_data['high'] - self.hist_data['low']) / self.hist_data['close']).replace(0, np.nan).dropna()
        self.hist_open_gap_ratio = (self.hist_data['open'] - self.hist_data['close'].shift(1)) / self.hist_data['close'].shift(1)
        self.hist_open_gap_ratio.dropna(inplace=True)
        self.hist_volume = self.hist_data['volume'].dropna()

    def fit_garch_model(self, p=1, q=1, dist='t'): # Changed 'Students-t' to 't'
        """Fits a GARCH(p, q) model to the historical returns."""
        logger.info(f"Fitting GARCH(p={p}, q={q}) model with '{dist}' distribution...")
        if self.hist_returns.empty:
            raise ValueError("Cannot fit GARCH model on empty returns series.")
        # Multiply returns by 100 as arch often expects percentage returns
        # Added rescale=False to suppress the DataScaleWarning.
        # If you want to explicitly rescale, remove rescale=False and adjust the multiplication factor (e.g., * 10000)
        garch_model = arch_model(self.hist_returns * 100, p=p, q=q, vol='Garch', dist=dist, rescale=False)
        self.fitted_model = garch_model.fit(disp='off')
        logger.info("GARCH model fitting complete.")

    def simulate_one_path(self, num_periods: int, start_date: pd.Timestamp, freq) -> Optional[pd.DataFrame]:
        """Generates a single, full synthetic OHLCV data path."""
        if self.fitted_model is None:
            logger.error("GARCH model not fitted. Call fit_garch_model() first.")
            return None

        # Get the last price and last conditional variance from the historical data for simulation start
        last_price = self.hist_data['close'].iloc[-1]
        # Convert variance back from percentage (if returns were multiplied by 100)
        # Changed [-1] to .iloc[-1] for positional indexing
        last_variance = self.fitted_model.conditional_volatility.iloc[-1]**2 / (100**2)
        
        # Extract GARCH parameters
        params = self.fitted_model.params
        # Adjust omega back to non-percentage scale if returns were scaled
        omega = params['omega'] / (100**2)
        alpha = params['alpha[1]']
        beta = params['beta[1]']
        nu = params.get('nu', np.inf) # Degrees of freedom for Students-t

        sim_returns = np.zeros(num_periods)
        current_variance = last_variance
        for t in range(num_periods):
            # Generate random shock based on the specified distribution
            # Corrected access to distribution name
            if self.fitted_model.model.distribution.name == 'StudentsT':
                # For Students-t, generate from standard t distribution and scale
                random_shock = np.random.standard_t(df=nu) * np.sqrt((nu - 2) / nu)
            else: # Assume Normal distribution if not Students-t
                random_shock = np.random.normal()
            
            # Generate return based on current conditional volatility
            sim_returns[t] = random_shock * np.sqrt(current_variance)
            
            # Update conditional variance for the next period
            current_variance = omega + alpha * (sim_returns[t]**2) + beta * current_variance
        
        # Calculate simulated close prices
        sim_close_prices = last_price * (1 + sim_returns).cumprod()

        # Create DataFrame for synthetic path
        synthetic_path = pd.DataFrame(index=pd.date_range(start=start_date, periods=num_periods, freq=freq))
        synthetic_path['close'] = sim_close_prices
        
        # Simulate 'open' prices based on historical open-close gaps
        # Need to handle the first 'prev_close' for the first simulated bar
        # Replaced .append() with pd.concat()
        prev_close = pd.concat([pd.Series([self.hist_data['close'].iloc[-1]]), synthetic_path['close'].iloc[:-1]], ignore_index=True)
        open_gaps = np.random.choice(self.hist_open_gap_ratio, size=num_periods)
        synthetic_path['open'] = prev_close.values * (1 + open_gaps)
        
        # Simulate 'high' and 'low' based on historical high-low ranges
        hl_range_ratios = np.random.choice(self.hist_hl_range_ratio.clip(0, 0.2), size=num_periods)
        hl_range = hl_range_ratios * synthetic_path['close']
        
        # Ensure high is always >= open and close, and low is always <= open and close
        synthetic_path['high'] = pd.concat([synthetic_path['open'], synthetic_path['close']], axis=1).max(axis=1) + hl_range * np.random.uniform(0, 1, size=num_periods)
        synthetic_path['low'] = pd.concat([synthetic_path['open'], synthetic_path['close']], axis=1).min(axis=1) - hl_range * np.random.uniform(0, 1, size=num_periods)
        
        # Final adjustment to ensure high >= open, close, low <= open, close
        synthetic_path['high'] = pd.concat([synthetic_path['high'], synthetic_path['open'], synthetic_path['close']], axis=1).max(axis=1)
        synthetic_path['low'] = pd.concat([synthetic_path['low'], synthetic_path['open'], synthetic_path['close']], axis=1).min(axis=1)
        
        # Simulate 'volume' based on historical volumes
        synthetic_path['volume'] = np.random.choice(self.hist_volume, size=num_periods)
        
        return synthetic_path[['open', 'high', 'low', 'close', 'volume']]


class MonteCarloAnalyzer:
    """Handles analysis, plotting, and saving of Monte Carlo results."""
    def __init__(self, metrics_df: pd.DataFrame, all_equity_curves: list, deterministic_results: dict, output_dir: Path):
        self.metrics_df = metrics_df
        self.all_equity_curves = all_equity_curves
        self.deterministic_results = deterministic_results
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("darkgrid")

    def save_summary_stats(self):
        """Calculates and saves descriptive statistics of performance metrics."""
        kpis = ['total_return_pct', 'max_drawdown_pct', 'win_rate_pct', 'profit_factor', 'num_trades', 'total_net_pnl']
        summary_stats = pd.DataFrame(index=['mean', 'std', 'min', '25%', '50%', '75%', 'max'])
        for kpi in kpis:
            if kpi in self.metrics_df.columns:
                # Ensure the column is numeric before describing
                numeric_col = pd.to_numeric(self.metrics_df[kpi], errors='coerce').dropna()
                if not numeric_col.empty:
                    summary_stats[kpi] = numeric_col.describe(percentiles=[.25, .5, .75]).loc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
                else:
                    summary_stats[kpi] = np.nan # Fill with NaN if no numeric data
            else:
                summary_stats[kpi] = np.nan # Fill with NaN if column not found
        
        filepath = self.output_dir / "1_performance_summary_stats.csv"
        summary_stats.to_csv(filepath)
        logger.info(f"Performance summary stats saved to {filepath}")
        logger.info("\n" + summary_stats.to_string(float_format="%.2f"))

        raw_filepath = self.output_dir / "all_simulation_metrics.csv"
        self.metrics_df.to_csv(raw_filepath, index=False)
        logger.info(f"Raw metrics for all simulations saved to {raw_filepath}")

    def plot_performance_distribution(self, metric='total_return_pct'):
        """Plots the distribution of a key performance metric."""
        # Convert metric column to numeric, coercing errors to NaN, then drop NaNs
        plot_data = pd.to_numeric(self.metrics_df[metric], errors='coerce').dropna()

        if plot_data.empty:
            logger.warning(f"Metric '{metric}' not available or contains no valid numeric data for plotting.")
            return

        plt.figure(figsize=(12, 7))
        sns.histplot(plot_data, kde=True, bins=30, stat="density")
        
        det_metric = self.deterministic_results['metrics'].get(metric)
        # Ensure deterministic metric is numeric and not NaN before plotting
        if pd.notna(det_metric) and isinstance(det_metric, (int, float)):
            plt.axvline(det_metric, color='red', linestyle='--', linewidth=2, label=f'Deterministic Result ({det_metric:.2f}%)')
        
        plt.title(f'Distribution of {metric.replace("_", " ").title()}', fontsize=16)
        plt.xlabel(metric.replace("_", " ").title())
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout() # Adjust layout to prevent labels overlapping
        
        filepath = self.output_dir / f"2_distribution_{metric}.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Distribution plot for '{metric}' saved to {filepath}")

    def plot_equity_curves(self, num_to_plot=50):
        """Plots a sample of simulated equity curves against the deterministic baseline."""
        plt.figure(figsize=(15, 8))
        
        # Plot a sample of simulated curves
        if self.all_equity_curves:
            # Filter out empty equity curves before sampling
            non_empty_equity_curves = [curve for curve in self.all_equity_curves if not curve.empty]
            if non_empty_equity_curves:
                sample_size = min(num_to_plot, len(non_empty_equity_curves))
                # Use random.sample for unique indices
                indices_to_plot = np.random.choice(len(non_empty_equity_curves), sample_size, replace=False)
                for i in indices_to_plot:
                    equity_curve = non_empty_equity_curves[i]
                    # Ensure index is DatetimeIndex for plotting
                    if isinstance(equity_curve.index, pd.DatetimeIndex):
                        plt.plot(equity_curve.index, equity_curve, alpha=0.2, linewidth=1)
                    else:
                        logger.warning(f"Skipping plotting of a simulated equity curve due to non-DatetimeIndex. Index type: {type(equity_curve.index)}")
            else:
                logger.warning("No non-empty simulated equity curves to plot.")


        # Plot the deterministic curve
        det_equity = self.deterministic_results.get('equity_curve')
        if det_equity is not None and not det_equity.empty:
            if isinstance(det_equity.index, pd.DatetimeIndex):
                plt.plot(det_equity.index, det_equity, color='red', linewidth=2.5, label='Deterministic Backtest')
            else:
                logger.warning(f"Skipping plotting deterministic equity curve due to non-DatetimeIndex. Index type: {type(det_equity.index)}")
        else:
            logger.warning("Deterministic equity curve is empty or not available for plotting.")
        
        plt.title('Simulated Equity Curves vs. Deterministic Backtest', fontsize=16)
        plt.xlabel('Date')
        plt.ylabel('Equity')
        plt.legend()
        plt.tight_layout()
        
        filepath = self.output_dir / "3_equity_curves_comparison.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Equity curve comparison plot saved to {filepath}")

    def plot_risk_reward_scatter(self, x_metric='max_drawdown_pct', y_metric='total_return_pct'):
        """Creates a scatter plot to visualize the risk/reward profile."""
        # Convert metrics columns to numeric, coercing errors to NaN, then drop NaNs
        x_data = pd.to_numeric(self.metrics_df[x_metric], errors='coerce')
        y_data = pd.to_numeric(self.metrics_df[y_metric], errors='coerce')
        
        plot_df = pd.DataFrame({x_metric: x_data, y_metric: y_data}).dropna()

        if plot_df.empty:
            logger.warning(f"Cannot create scatter plot; no valid numeric data for {x_metric} or {y_metric}.")
            return
            
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=plot_df, x=x_metric, y=y_metric, alpha=0.6)
        
        det_x = self.deterministic_results['metrics'].get(x_metric)
        det_y = self.deterministic_results['metrics'].get(y_metric)
        # Ensure deterministic metrics are numeric and not NaN before plotting
        if pd.notna(det_x) and pd.notna(det_y) and isinstance(det_x, (int, float)) and isinstance(det_y, (int, float)):
            plt.scatter(det_x, det_y, color='red', s=150, marker='*', label='Deterministic Result', zorder=5)

        plt.title('Risk vs. Reward Profile (Each point is one simulation)', fontsize=16)
        plt.xlabel(x_metric.replace("_", " ").title())
        plt.ylabel(y_metric.replace("_", " ").title())
        plt.legend()
        plt.tight_layout()
        
        filepath = self.output_dir / "4_risk_reward_scatter.png"
        plt.savefig(filepath, dpi=300)
        plt.close()
        logger.info(f"Risk/reward scatter plot saved to {filepath}")

    def run_full_analysis(self):
        """Runs all analysis and plotting steps."""
        logger.info("--- Starting Monte Carlo Results Analysis ---")
        self.save_summary_stats()
        self.plot_performance_distribution(metric='total_return_pct')
        self.plot_equity_curves()
        self.plot_risk_reward_scatter()
        logger.info("--- Monte Carlo Results Analysis Complete ---")


def run_mc_backtest_pipeline(symbol: str, interval: str, model_key: str, backtest_mode: str, train_ratio: float, num_simulations: int):
    logger.info(f"\n--- Starting Monte Carlo Backtest Run ---")
    logger.info(f"Symbol: {symbol}, Interval: {interval}, Model: {model_key}, Mode: {backtest_mode}")
    logger.info(f"Simulations: {num_simulations}")

    # --- 1. Load and Split Data ---
    dm = DataManager()
    try:
        data = dm.load_data(symbol=symbol, interval=interval, data_type='processed')
        if data is None or data.empty:
            raise FileNotFoundError(f"Historical processed data not found for {symbol} {interval}.")
        
        # Ensure data index is DatetimeIndex and has a frequency
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index, utc=True)
        if data.index.freq is None:
            # Attempt to infer frequency, fallback to a default if not possible
            inferred_freq = pd.infer_freq(data.index)
            if inferred_freq:
                data.index.freq = inferred_freq
                logger.info(f"Inferred data frequency: {inferred_freq}")
            else:
                logger.warning("Could not infer data frequency. Using 'min' as a fallback. This might cause issues with date_range generation.")
                data.index.freq = 'min' # Fallback to minute frequency

        if backtest_mode == 'full':
            train_data, test_data = data, data
        else:
            train_size = int(len(data) * train_ratio)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
        
        num_periods = len(test_data)
        logger.info(f"Data split: {len(train_data)} train bars, {len(test_data)} test bars.")
        if num_periods == 0:
            raise ValueError("Test data set is empty. Adjust train_ratio or data size.")

    except Exception as e:
        logger.critical(f"Data loading/splitting failed: {e}", exc_info=True)
        sys.exit(1)

    # --- 2. Run Deterministic Backtest for Baseline ---
    logger.info("--- Running Deterministic Backtest for Baseline ---")
    deterministic_results = {}
    try:
        model_specific_config = MODEL_CONFIG.get(model_key)
        if model_specific_config is None:
            raise ValueError(f"Model configuration for '{model_key}' not found in MODEL_CONFIG.")

        trainer = ModelTrainer(config=model_specific_config.copy())
        trainer.load(symbol=symbol, interval=interval, model_key=model_key)
        
        # Clean test_data features before prediction, similar to backtest.py
        # This step is crucial to ensure consistency
        model_feature_cols = trainer.feature_columns_original
        if not model_feature_cols:
            logger.warning("Original feature columns not found in loaded model metadata. Using features_to_use from config as fallback.")
            model_feature_cols = model_specific_config.get('features_to_use', [])
            if not model_feature_cols:
                raise RuntimeError("Could not determine original feature columns used by the model.")

        # Drop rows with NaNs in any of the model's feature columns in test_data
        initial_test_data_len = len(test_data)
        test_data_cleaned = test_data.dropna(subset=model_feature_cols).copy()
        rows_removed_cleaning = initial_test_data_len - len(test_data_cleaned)
        if rows_removed_cleaning > 0:
            logger.info(f"Removed {rows_removed_cleaning} rows from deterministic test_data due to NaNs in model feature columns.")
        
        if test_data_cleaned.empty:
            logger.warning("Deterministic test data is empty after cleaning NaNs. Skipping deterministic backtest.")
            # Create empty results to allow MC loop to proceed
            deterministic_results = {'trades': pd.DataFrame(), 'equity_curve': pd.Series(dtype=float), 'metrics': {}}
        else:
            predictions = trainer.predict(test_data_cleaned).reindex(test_data_cleaned.index).fillna(0).astype(int)
            probabilities = trainer.predict_proba(test_data_cleaned).reindex(test_data_cleaned.index)

            det_backtester = Backtester(
                data=test_data_cleaned.copy(), # Pass cleaned data
                model_predict=predictions, model_proba=probabilities,
                symbol=symbol, interval=interval, model_type=model_key,
                backtester_config_override=copy.deepcopy(BACKTESTER_CONFIG),
                strategy_config_override=copy.deepcopy(STRATEGY_CONFIG),
                paths_override=copy.deepcopy(PATHS)
            )
            # Run with saving enabled to get the standard deterministic result
            det_trades, det_equity, det_metrics = det_backtester.run_backtest()
            deterministic_results = {'trades': det_trades, 'equity_curve': det_equity, 'metrics': det_metrics}
            logger.info("Deterministic backtest complete.")
    except Exception as e:
        logger.critical(f"Deterministic backtest failed, cannot proceed: {e}", exc_info=True)
        sys.exit(1)

    # --- 3. Initialize and Fit GARCH Simulator ---
    # Ensure train_data has enough history for GARCH fitting (e.g., at least 2 for pct_change)
    if train_data.empty or len(train_data) < 2:
        logger.critical("Train data is too short or empty for GARCH model fitting. Exiting.")
        sys.exit(1)

    simulator = GarchOHLCVSimulator(train_data)
    try:
        simulator.fit_garch_model()
    except Exception as e:
        logger.critical(f"GARCH model fitting failed: {e}", exc_info=True)
        sys.exit(1)

    # --- 4. Run Simulation Loop ---
    all_metrics, all_equity_curves = [], []
    logger.info("--- Starting Monte Carlo Simulation Loop ---")
    
    # Define a template for expected metrics to ensure consistency
    # This should match the keys produced by Backtester._calculate_summary_metrics
    metric_keys_template = [
        'initial_capital', 'final_equity', 'total_return_pct', 'peak_equity',
        'max_drawdown_pct', 'equity_change', 'num_trades', 'total_net_pnl',
        'total_gross_pnl', 'total_fees', 'num_wins', 'num_losses',
        'win_rate_pct', 'avg_pnl_per_trade', 'avg_win_pnl', 'avg_loss_pnl',
        'profit_factor', 'avg_holding_duration_bars', 'config_symbol',
        'config_interval', 'config_model_type', 'config_leverage',
        'config_risk_per_trade_pct', 'config_trading_fee_rate',
        'config_maintenance_margin_rate', 'config_liquidation_fee_rate',
        'config_volatility_adjustment_enabled', 'config_volatility_window_bars',
        'config_fixed_take_profit_pct', 'config_fixed_stop_loss_pct',
        'config_alpha_take_profit', 'config_alpha_stop_loss',
        'config_trend_filter_enabled', 'config_trend_filter_ema_period',
        'config_default_max_holding_period_bars',
        'config_volatility_regime_filter_enabled',
        'config_volatility_regime_max_holding_bars',
        'config_allow_trading_in_volatility_regime', 'config_min_quantity',
        'config_min_notional', 'config_tie_breaker', 'config_exit_on_neutral_signal',
        'config_allow_long_trades', 'config_allow_short_trades',
        'config_confidence_filter_enabled', 'config_confidence_threshold_long_pct',
        'config_confidence_threshold_short_pct', 'PnL Consistency Check'
    ]

    # Pre-load ModelTrainer for simulations to avoid re-loading in each loop
    # The trainer instance is already loaded from the deterministic step, reuse it.
    
    for i in tqdm(range(num_simulations), desc="Running Backtest Simulations"):
        # Initialize a default metrics dictionary for this simulation
        sim_summary_metrics = {k: np.nan for k in metric_keys_template}
        sim_equity_curve = pd.Series(dtype=float) # Initialize as empty

        try:
            # Generate synthetic data path
            synthetic_df = simulator.simulate_one_path(num_periods, start_date=test_data.index[0], freq=test_data.index.freq)
            if synthetic_df is None or synthetic_df.empty:
                logger.warning(f"Simulation {i+1}: Generated empty synthetic data. Skipping.")
                sim_summary_metrics['error'] = 'Empty synthetic data'
                all_metrics.append(sim_summary_metrics)
                all_equity_curves.append(sim_equity_curve)
                continue
            
            # Feature engineer the synthetic data
            feature_engineer = FeaturesEngineer(config=copy.deepcopy(FEATURE_CONFIG))
            featured_df = feature_engineer.process(synthetic_df)
            
            # Clean features in the synthetic data before prediction, using the same feature columns as the model
            featured_df_cleaned = featured_df.dropna(subset=model_feature_cols).copy()
            if featured_df_cleaned.empty:
                logger.warning(f"Simulation {i+1}: Featured data is empty after cleaning NaNs. Skipping.")
                sim_summary_metrics['error'] = 'Empty featured data after cleaning'
                all_metrics.append(sim_summary_metrics)
                all_equity_curves.append(sim_equity_curve)
                continue

            # Generate predictions and probabilities using the *pre-loaded* trainer
            sim_predictions = trainer.predict(featured_df_cleaned).reindex(featured_df_cleaned.index).fillna(0).astype(int)
            sim_probabilities = trainer.predict_proba(featured_df_cleaned).reindex(featured_df_cleaned.index)

            # Initialize and run backtester for the simulation
            sim_backtester = Backtester(
                data=featured_df_cleaned.copy(), # Pass cleaned data
                model_predict=sim_predictions, model_proba=sim_probabilities,
                symbol=symbol, interval=interval, model_type=f"{model_key}_mc_sim",
                backtester_config_override=copy.deepcopy(BACKTESTER_CONFIG),
                strategy_config_override=copy.deepcopy(STRATEGY_CONFIG)
            )
            # Disable saving results for individual simulations to save disk space and time
            sim_backtester.save_trades = sim_backtester.save_equity_curve = sim_backtester.save_metrics = False
            
            _, sim_equity_curve, temp_summary_metrics = sim_backtester.run_backtest()
            
            # Update the pre-initialized sim_summary_metrics with actual results
            sim_summary_metrics.update(temp_summary_metrics)

            all_metrics.append(sim_summary_metrics)
            all_equity_curves.append(sim_equity_curve)
        except Exception as e:
            logger.error(f"Backtest on simulation {i+1} failed: {e}", exc_info=False)
            sim_summary_metrics['error'] = str(e)
            all_metrics.append(sim_summary_metrics)
            all_equity_curves.append(sim_equity_curve) # Append empty or partially filled equity curve


    # --- 5. Aggregate and Analyze Results ---
    if not all_metrics:
        logger.error("No simulations were successfully completed. Exiting.")
        return

    # Define a safe, dedicated output directory that aligns with project structure
    # The results are a form of analysis on a backtest for a specific model
    base_analysis_dir = Path(PATHS.get("backtesting_analysis_dir"))
    # Create a model-specific subfolder, then a subfolder for the MC analysis itself
    output_dir = base_analysis_dir / model_key / f"{symbol.replace('/', '_')}_{interval}_monte_carlo"

    analyzer = MonteCarloAnalyzer(pd.DataFrame(all_metrics), all_equity_curves, deterministic_results, output_dir)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run advanced Monte Carlo backtests on a trained model.")
    parser.add_argument('--symbol', type=str, required=True, help='Trading pair symbol (e.g., BTCUSDT)')
    parser.add_argument('--interval', type=str, required=True, help='Time interval (e.g., 1h, 1d)')
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_CONFIG.keys()), help='Model key from MODEL_CONFIG')
    parser.add_argument('--backtest_mode', type=str, default='test', choices=['full', 'train', 'test'], help='Data split to use for GARCH fitting and simulation length.')
    parser.add_argument('--train_ratio', type=float, default=GENERAL_CONFIG.get('train_test_split_ratio', 0.8), help='Train/test split ratio.')
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
        logging.shutdown()

