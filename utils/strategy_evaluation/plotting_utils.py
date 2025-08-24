# utils/strategy_evaluation/plotting_utils.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, Any, List, Optional

from config.params import FLOAT_EPSILON

logger = logging.getLogger(__name__)

class PlottingUtils:
    """
    A stateless utility class for creating performance analysis plots.
    Each method generates and returns a Matplotlib Figure object without saving it.
    It supports plotting for both single-run and Monte Carlo analyses.
    """
    def __init__(self):
        """Initializes the PlottingUtils."""
        self.logger = logging.getLogger(self.__class__.__name__)
        sns.set_style("whitegrid")

    def plot_equity_curve(self, equity_df: pd.DataFrame, title: str) -> plt.Figure:
        """Generates a plot of the equity curve."""
        self.logger.info("Generating equity curve plot figure...")
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if not equity_df.empty and 'equity' in equity_df.columns:
            ax.plot(equity_df.index, equity_df['equity'], label='Equity Curve', color='blue')
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Equity (USD)', fontsize=12)
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'No equity data available', ha='center', va='center')
            ax.set_title(title, fontsize=16)

        fig.tight_layout()
        return fig

    def plot_drawdown_curve(self, equity_df: pd.DataFrame, title: str) -> plt.Figure:
        """Generates a plot of the portfolio's drawdown curve."""
        self.logger.info("Generating drawdown curve plot figure...")
        fig, ax = plt.subplots(figsize=(12, 7))

        if not equity_df.empty and 'equity' in equity_df.columns:
            rolling_max = equity_df['equity'].cummax()
            drawdown = (equity_df['equity'] - rolling_max) / (rolling_max + FLOAT_EPSILON)
            drawdown_pct = drawdown * 100

            ax.fill_between(drawdown_pct.index, drawdown_pct, 0, color='red', alpha=0.3)
            ax.plot(drawdown_pct.index, drawdown_pct, label='Drawdown', color='red', linewidth=1.5)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Drawdown (%)', fontsize=12)
            ax.legend()
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, 'No equity data available for drawdown', ha='center', va='center')
            ax.set_title(title, fontsize=16)
        
        fig.tight_layout()
        return fig

    def plot_trade_pnl_distribution(self, trade_history_df: pd.DataFrame, title: str) -> plt.Figure:
        """Generates a histogram of the Profit and Loss (PnL) for all trades."""
        self.logger.info("Generating trade PnL distribution plot figure...")
        fig, ax = plt.subplots(figsize=(12, 7))

        if not trade_history_df.empty and 'net_pnl' in trade_history_df.columns:
            sns.histplot(trade_history_df['net_pnl'], kde=True, ax=ax, bins=50)
            ax.axvline(trade_history_df['net_pnl'].mean(), color='red', linestyle='--', label=f"Mean PnL: ${trade_history_df['net_pnl'].mean():.2f}")
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Net PnL (USD)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No trade data available for PnL distribution', ha='center', va='center')
            ax.set_title(title, fontsize=16)

        fig.tight_layout()
        return fig
        
    def plot_exit_reason_pnl_boxplot(self, trade_history_df: pd.DataFrame, title: str) -> plt.Figure:
        """Generates a boxplot of PnL grouped by the reason for exiting the trade."""
        self.logger.info("Generating PnL by exit reason boxplot figure...")
        fig, ax = plt.subplots(figsize=(12, 7))

        if not trade_history_df.empty and 'exit_reason' in trade_history_df.columns and 'net_pnl' in trade_history_df.columns:
            sns.boxplot(x='exit_reason', y='net_pnl', data=trade_history_df, ax=ax)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Exit Reason', fontsize=12)
            ax.set_ylabel('Net PnL (USD)', fontsize=12)
            plt.xticks(rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No data for exit reason analysis', ha='center', va='center')
            ax.set_title(title, fontsize=16)

        fig.tight_layout()
        return fig

    def plot_exit_reason_frequency_barplot(self, trade_history_df: pd.DataFrame, title: str) -> plt.Figure:
        """Generates a barplot showing the frequency of each exit reason."""
        self.logger.info("Generating exit reason frequency barplot figure...")
        fig, ax = plt.subplots(figsize=(12, 7))

        if not trade_history_df.empty and 'exit_reason' in trade_history_df.columns:
            exit_counts = trade_history_df['exit_reason'].value_counts()
            sns.barplot(x=exit_counts.index, y=exit_counts.values, ax=ax)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Exit Reason', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            plt.xticks(rotation=45, ha='right')
        else:
            ax.text(0.5, 0.5, 'No data for exit reason frequency', ha='center', va='center')
            ax.set_title(title, fontsize=16)
        
        fig.tight_layout()
        return fig

    def plot_performance_distribution(self, metrics_df: pd.DataFrame, metric: str, deterministic_metric_value: Optional[float]) -> plt.Figure:
        """Plots the distribution of a key performance metric from multiple simulations."""
        self.logger.info(f"Generating performance distribution plot figure for metric: {metric}...")
        fig, ax = plt.subplots(figsize=(12, 7))
        plot_data = pd.to_numeric(metrics_df.get(metric), errors='coerce').dropna()

        if not plot_data.empty:
            sns.histplot(plot_data, kde=True, bins=30, stat="density", ax=ax, label='Simulations', edgecolor='black', alpha=0.6)
            # --- ROBUSTNESS FIX: Check for both None and NaN ---
            if deterministic_metric_value is not None and pd.notna(deterministic_metric_value):
                ax.axvline(deterministic_metric_value, color='red', linestyle='--', linewidth=2.5, label=f'Deterministic: {deterministic_metric_value:.2f}')
            ax.set_title(f'Distribution of {metric}', fontsize=16)
            ax.set_xlabel(metric, fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend()
        else:
            ax.text(0.5, 0.5, f"No simulation data for metric '{metric}'", ha='center', va='center')
            ax.set_title(f'Distribution of {metric}', fontsize=16)

        fig.tight_layout()
        return fig

    def plot_equity_curves_comparison(self, all_equity_curves: List[pd.Series], deterministic_equity: Optional[pd.DataFrame], num_to_plot: int) -> plt.Figure:
        """Plots a sample of simulated equity curves against a deterministic baseline."""
        self.logger.info("Generating equity curves comparison plot figure...")
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # --- ROBUSTNESS FIX: Ensure deterministic_equity is a DataFrame and has the 'equity' column ---
        has_det_equity = isinstance(deterministic_equity, pd.DataFrame) and not deterministic_equity.empty and 'equity' in deterministic_equity.columns
        
        non_empty_sim_curves = [curve for curve in all_equity_curves if isinstance(curve, pd.Series) and not curve.empty]
        if non_empty_sim_curves:
            sample_size = min(num_to_plot, len(non_empty_sim_curves))
            indices_to_plot = np.random.choice(len(non_empty_sim_curves), sample_size, replace=False)
            for i, idx in enumerate(indices_to_plot):
                label = 'Simulated Runs' if i == 0 else None
                ax.plot(non_empty_sim_curves[idx].index, non_empty_sim_curves[idx], alpha=0.3, linewidth=1.5, label=label, color='sandybrown')

        if has_det_equity:
            ax.plot(deterministic_equity.index, deterministic_equity['equity'], color='red', linewidth=2.5, label='Deterministic Backtest')
        
        ax.set_title('Simulated Equity Curves vs. Deterministic Backtest', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity', fontsize=12)
        if non_empty_sim_curves or has_det_equity:
            ax.legend()
        fig.tight_layout()
        return fig

    def plot_risk_reward_scatter(self, metrics_df: pd.DataFrame, x_metric: str, y_metric: str, deterministic_x: Optional[float], deterministic_y: Optional[float]) -> plt.Figure:
        """Creates a scatter plot to visualize the risk/reward profile."""
        self.logger.info(f"Generating risk/reward scatter plot figure ({y_metric} vs {x_metric})...")
        fig, ax = plt.subplots(figsize=(10, 8))
        x_data = pd.to_numeric(metrics_df.get(x_metric), errors='coerce')
        y_data = pd.to_numeric(metrics_df.get(y_metric), errors='coerce')
        
        plot_df = pd.DataFrame({x_metric: x_data, y_metric: y_data}).dropna()

        if not plot_df.empty:
            sns.scatterplot(data=plot_df, x=x_metric, y=y_metric, alpha=0.6, ax=ax, label='Simulations', s=50)
            # --- ROBUSTNESS FIX: Check for both None and NaN for both coordinates ---
            if pd.notna(deterministic_x) and pd.notna(deterministic_y):
                ax.scatter(deterministic_x, deterministic_y, color='red', s=200, marker='*', label='Deterministic Result', zorder=5, edgecolor='black')
            ax.set_title('Risk vs. Reward Profile (Each point is one simulation)', fontsize=16)
            ax.set_xlabel(x_metric, fontsize=12)
            ax.set_ylabel(y_metric, fontsize=12)
            ax.grid(True)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No simulation data for risk/reward scatter', ha='center', va='center')
            ax.set_title('Risk vs. Reward Profile', fontsize=16)
        
        fig.tight_layout()
        return fig

    def plot_simulated_ohlcv_paths(self, all_simulated_paths: List[pd.DataFrame], deterministic_ohlcv: Optional[pd.DataFrame], num_to_plot: int) -> plt.Figure:
        """Plots a sample of simulated OHLCV paths (close prices)."""
        self.logger.info("Generating simulated OHLCV paths plot figure...")
        fig, ax = plt.subplots(figsize=(15, 8))
        
        has_det_ohlcv = isinstance(deterministic_ohlcv, pd.DataFrame) and not deterministic_ohlcv.empty and 'close' in deterministic_ohlcv.columns
        
        non_empty_paths = [path for path in all_simulated_paths if isinstance(path, pd.DataFrame) and not path.empty]
        if non_empty_paths:
            sample_size = min(num_to_plot, len(non_empty_paths))
            indices_to_plot = np.random.choice(len(non_empty_paths), sample_size, replace=False)
            for i, idx in enumerate(indices_to_plot):
                label = 'Simulated Paths' if i == 0 else None
                ax.plot(non_empty_paths[idx].index, non_empty_paths[idx]['close'], alpha=0.5, linewidth=1.0, label=label, color='grey')

        if has_det_ohlcv:
            ax.plot(deterministic_ohlcv.index, deterministic_ohlcv['close'], color='black', linewidth=2.0, linestyle='--', label='Actual Test Data Close')

        ax.set_title('Sample of Simulated OHLCV Paths vs. Actual Test Data', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price', fontsize=12)
        if non_empty_paths or has_det_ohlcv:
            ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig