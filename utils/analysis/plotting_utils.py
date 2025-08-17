# utils/analysis/plotting_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import FLOAT_EPSILON for consistency in plotting calculations if needed
from config.params import FLOAT_EPSILON
from config.paths import PATHS # For plot pattern


logger = logging.getLogger(__name__)

class PlottingUtils:
    """
    Provides utility methods for generating various performance plots.
    Designed to be used by both PerformanceAnalyzer and MonteCarloAnalyzer
    to centralize plotting logic and ensure consistent visualizations.
    """
    def __init__(self, symbol: str, interval: str, model_type: str, analysis_dir: Path):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.symbol = symbol.upper()
        self.interval = interval
        self.model_type = model_type
        self.analysis_dir = analysis_dir
        
        # Ensure the analysis directory exists
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"PlottingUtils initialized. Plots will be saved to: {self.analysis_dir}")


    def _save_plot(self, fig: plt.Figure, plot_type: str):
        """Helper to save a matplotlib figure to the designated analysis directory."""
        plot_context = f"{self.symbol}_{self.interval}_{self.model_type}"
        plot_filename = PATHS.get("analysis_plot_pattern", "{symbol}_{interval}_{model_type}_{analysis_type}.png").format(
            symbol=self.symbol, interval=self.interval, model_type=self.model_type, analysis_type=plot_type)
        save_path = self.analysis_dir / plot_filename
        
        try:
            fig.tight_layout() # Ensure tight layout before saving
            fig.savefig(save_path, dpi=300)
            self.logger.debug(f"Plot '{plot_type}' saved to {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save plot '{plot_type}' to {save_path}: {e}", exc_info=True)
        finally:
            plt.close(fig) # Close the figure to free up memory

    def plot_equity_curve(self, equity_df: pd.DataFrame, title_suffix: str = ""):
        """Plots the equity curve over time."""
        if equity_df.empty or 'equity' not in equity_df.columns or equity_df['equity'].empty:
            self.logger.warning("Equity data is empty or missing 'equity' column. Skipping equity curve plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Ensure the index is datetime for plotting
        if not isinstance(equity_df.index, pd.DatetimeIndex):
            equity_df.index = pd.to_datetime(equity_df.index, errors='coerce', utc=True)
            equity_df.dropna(subset=[equity_df.index.name if equity_df.index.name else equity_df.index], inplace=True) # handle unnamed index

        ax.plot(equity_df.index, equity_df['equity'], label='Equity Curve', color='blue')
        ax.set_title(f'Equity Curve - {self.symbol} {self.interval} {self.model_type} {title_suffix}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Account Equity')
        ax.grid(True)
        ax.legend()
        self._save_plot(fig, "equity_curve")

    def plot_drawdown_curve(self, equity_df: pd.DataFrame, title_suffix: str = ""):
        """Plots the drawdown curve over time."""
        if equity_df.empty or 'equity' not in equity_df.columns or equity_df['equity'].empty:
            self.logger.warning("Equity data is empty or missing 'equity' column. Skipping drawdown curve plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        rolling_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - rolling_max) / (rolling_max + FLOAT_EPSILON) * 100 # In percentage
        
        # Ensure the index is datetime for plotting
        if not isinstance(drawdown.index, pd.DatetimeIndex):
            drawdown.index = pd.to_datetime(drawdown.index, errors='coerce', utc=True)
            drawdown.dropna(subset=[drawdown.index.name if drawdown.index.name else drawdown.index], inplace=True) # handle unnamed index


        ax.plot(drawdown.index, drawdown, label='Drawdown (%)', color='red')
        ax.set_title(f'Drawdown Curve - {self.symbol} {self.interval} {self.model_type} {title_suffix}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        ax.fill_between(drawdown.index, drawdown, color='red', alpha=0.3)
        ax.legend()
        self._save_plot(fig, "drawdown_curve")

    def plot_trade_pnl_distribution(self, trade_history_df: pd.DataFrame, title_suffix: str = ""):
        """Plots a histogram of trade Net PnL."""
        if trade_history_df.empty or 'net_pnl' not in trade_history_df.columns or trade_history_df['net_pnl'].dropna().empty:
            self.logger.warning("Trade net PnL data is not available or empty. Skipping PnL distribution plot.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(trade_history_df['net_pnl'], bins=50, kde=True, ax=ax)
        ax.set_title(f'Distribution of Net PnL per Trade - {self.symbol} {self.interval} {self.model_type} {title_suffix}')
        ax.set_xlabel('Net PnL')
        ax.set_ylabel('Frequency')
        ax.grid(True, axis='y')
        self._save_plot(fig, "pnl_distribution")

    def plot_exit_reason_pnl_boxplot(self, trade_history_df: pd.DataFrame, title_suffix: str = ""):
        """
        Generates a box plot showing Net PnL distribution by exit reason.
        """
        if trade_history_df.empty or 'net_pnl' not in trade_history_df.columns or 'exit_reason' not in trade_history_df.columns or trade_history_df.dropna(subset=['net_pnl', 'exit_reason']).empty:
            self.logger.warning("No valid trade data with exit reasons for PnL by exit reason plot. Skipping.")
            return

        self.logger.info("Generating Net PnL distribution plot by exit reason...")
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_data = trade_history_df.dropna(subset=['net_pnl', 'exit_reason']).copy()
        
        order = plot_data.groupby('exit_reason')['net_pnl'].median().sort_values(ascending=False).index
        sns.boxplot(data=plot_data, x='exit_reason', y='net_pnl', order=order, ax=ax)

        ax.set_title(f'Net PnL Distribution by Exit Reason - {self.symbol} {self.interval} {self.model_type} {title_suffix}')
        ax.set_xlabel("Exit Reason")
        ax.set_ylabel("Net PnL per Trade")
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, axis='y')
        self._save_plot(fig, "pnl_by_exit_reason")

    def plot_exit_reason_frequency_barplot(self, trade_history_df: pd.DataFrame, title_suffix: str = ""):
        """
        Generates a bar plot showing the frequency of trades per exit reason.
        """
        if trade_history_df.empty or 'exit_reason' not in trade_history_df.columns or trade_history_df['exit_reason'].dropna().empty:
            self.logger.warning("Trade data is missing 'exit_reason' for exit reason frequency plot or is empty. Skipping.")
            return

        self.logger.info("Generating trade frequency plot by exit reason...")
        fig, ax = plt.subplots(figsize=(10, 6))
        exit_reason_counts = trade_history_df['exit_reason'].value_counts().reset_index()
        exit_reason_counts.columns = ['exit_reason', 'count']

        order = exit_reason_counts.sort_values(by='count', ascending=False)['exit_reason']
        sns.barplot(data=exit_reason_counts, x='exit_reason', y='count', order=order, ax=ax)

        ax.set_title(f'Trade Frequency by Exit Reason - {self.symbol} {self.interval} {self.model_type} {title_suffix}')
        ax.set_xlabel("Exit Reason")
        ax.set_ylabel("Number of Trades")
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, axis='y')
        self._save_plot(fig, "exit_reason_frequency")

    def plot_performance_distribution(self, metrics_df: pd.DataFrame, metric: str = 'Total Return (%)'):
        """Plots the distribution of a key performance metric from multiple simulations."""
        plot_data = pd.to_numeric(metrics_df[metric], errors='coerce').dropna()

        if plot_data.empty:
            self.logger.warning(f"Metric '{metric}' not available or contains no valid numeric data for plotting.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.histplot(plot_data, kde=True, bins=30, stat="density", ax=ax)
        
        # Deterministic result will be added by the MonteCarloAnalyzer directly if needed
        
        ax.set_title(f'Distribution of {metric}', fontsize=16)
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.legend()
        self._save_plot(fig, f"distribution_{metric.replace(' (%)', '').replace(' ', '_').lower()}")

    def plot_equity_curves_comparison(self, all_equity_curves: List[pd.Series], deterministic_equity: Optional[pd.Series], num_to_plot: int = 50):
        """Plots a sample of simulated equity curves against a deterministic baseline."""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        if all_equity_curves:
            non_empty_sim_curves = [curve for curve in all_equity_curves if not curve.empty]
            if non_empty_sim_curves:
                sample_size = min(num_to_plot, len(non_empty_sim_curves))
                indices_to_plot = np.random.choice(len(non_empty_sim_curves), sample_size, replace=False)
                for i in indices_to_plot:
                    equity_curve = non_empty_sim_curves[i]
                    if isinstance(equity_curve.index, pd.DatetimeIndex):
                        ax.plot(equity_curve.index, equity_curve, alpha=0.2, linewidth=1)
                    else:
                        self.logger.warning(f"Skipping plotting of a simulated equity curve due to non-DatetimeIndex. Index type: {type(equity_curve.index)}")
            else:
                self.logger.warning("No non-empty simulated equity curves to plot.")


        if deterministic_equity is not None and not deterministic_equity.empty:
            if isinstance(deterministic_equity.index, pd.DatetimeIndex):
                ax.plot(deterministic_equity.index, deterministic_equity, color='red', linewidth=2.5, label='Deterministic Backtest')
            else:
                self.logger.warning(f"Skipping plotting deterministic equity curve due to non-DatetimeIndex. Index type: {type(deterministic_equity.index)}")
        else:
            self.logger.warning("Deterministic equity curve is empty or not available for plotting.")
        
        ax.set_title('Simulated Equity Curves vs. Deterministic Backtest', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.legend()
        self._save_plot(fig, "equity_curves_comparison")


    def plot_risk_reward_scatter(self, metrics_df: pd.DataFrame, x_metric: str = 'Max Drawdown (%)', y_metric: str = 'Total Return (%)', deterministic_x: Optional[float] = None, deterministic_y: Optional[float] = None):
        """Creates a scatter plot to visualize the risk/reward profile."""
        x_data = pd.to_numeric(metrics_df[x_metric], errors='coerce')
        y_data = pd.to_numeric(metrics_df[y_metric], errors='coerce')
        
        plot_df = pd.DataFrame({x_metric: x_data, y_metric: y_data}).dropna()

        if plot_df.empty:
            self.logger.warning(f"Cannot create scatter plot; no valid numeric data for {x_metric} or {y_metric}.")
            return
            
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=plot_df, x=x_metric, y=y_data, alpha=0.6, ax=ax)
        
        if pd.notna(deterministic_x) and pd.notna(deterministic_y) and isinstance(deterministic_x, (int, float)) and isinstance(deterministic_y, (int, float)):
            ax.scatter(deterministic_x, deterministic_y, color='red', s=150, marker='*', label='Deterministic Result', zorder=5)

        ax.set_title('Risk vs. Reward Profile (Each point is one simulation)', fontsize=16)
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.grid(True)
        ax.legend()
        self._save_plot(fig, "risk_reward_scatter")

    def plot_simulated_ohlcv_paths(self, all_simulated_paths: List[pd.DataFrame], deterministic_ohlcv: Optional[pd.DataFrame], num_to_plot: int = 5):
        """
        Plots a sample of simulated OHLCV paths (close prices) for visual inspection.
        """
        if not all_simulated_paths:
            self.logger.warning("No simulated OHLCV paths available to plot.")
            return

        fig, ax = plt.subplots(figsize=(15, 8))
        
        non_empty_paths = [path for path in all_simulated_paths if not path.empty]

        if not non_empty_paths:
            self.logger.warning("All simulated OHLCV paths are empty. Skipping plot.")
            self._save_plot(fig, "simulated_ohlcv_paths_empty") # Save an empty plot if all paths are empty
            return

        sample_size = min(num_to_plot, len(non_empty_paths))
        indices_to_plot = np.random.choice(len(non_empty_paths), sample_size, replace=False)

        for i in indices_to_plot:
            simulated_path = non_empty_paths[i]
            if isinstance(simulated_path.index, pd.DatetimeIndex):
                ax.plot(simulated_path.index, simulated_path['close'], alpha=0.6, linewidth=1.5, label=f'Sim {i+1} Close')
            else:
                self.logger.warning(f"Skipping plotting of simulated OHLCV path {i+1} due to non-DatetimeIndex.")

        if deterministic_ohlcv is not None and not deterministic_ohlcv.empty and 'close' in deterministic_ohlcv.columns:
            if isinstance(deterministic_ohlcv.index, pd.DatetimeIndex):
                ax.plot(deterministic_ohlcv.index, deterministic_ohlcv['close'], color='black', linewidth=2.0, linestyle='--', label='Actual Test Data Close')
            else:
                self.logger.warning(f"Skipping plotting actual test data OHLCV due to non-DatetimeIndex.")


        ax.set_title(f'Sample of Simulated OHLCV Paths (Close Price) vs. Actual Test Data\n{self.symbol} {self.interval}', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        self._save_plot(fig, "simulated_ohlcv_paths")

