# utils/labeling/analysis_plotter.py

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

class AnalysisPlotter:
    """
    Creates plots related to label analysis and returns matplotlib Figure objects.
    This class does NOT save files. It is a pure plotting utility.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.logger.debug("AnalysisPlotter initialized.")

    def plot_label_distribution(self, distribution_summary: pd.DataFrame, **kwargs) -> 'Figure':
        """Generates a bar plot of the label distribution and returns the figure."""
        symbol = kwargs.get('symbol', '')
        interval = kwargs.get('interval', '')
        self.logger.debug(f"Creating Label Distribution plot for {symbol} {interval}...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x='Label', y='Percentage', data=distribution_summary, palette='viridis', hue='Label', legend=False, ax=ax)
        ax.set_title(f'Overall Label Distribution (%)\n{symbol} {interval}')
        ax.set_xlabel('Label (-1: Short, 0: Neutral, 1: Long)')
        ax.set_ylabel('Percentage (%)')
        ax.grid(axis='y', linestyle='--')
        fig.tight_layout()
        return fig

    def plot_net_return_distributions(self, df_net_returns: pd.DataFrame, **kwargs) -> 'Figure':
        """Generates histograms of Net Return distributions and returns the figure."""
        symbol = kwargs.get('symbol', '')
        interval = kwargs.get('interval', '')
        self.logger.debug(f"Creating Net Return Distributions plot for {symbol} {interval}...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        long_returns = df_net_returns['Net_Return_Long'].dropna()
        if not long_returns.empty:
            p_low, p_high = long_returns.quantile([0.005, 0.995])
            sns.histplot(data=long_returns[long_returns.between(p_low, p_high)], bins=100, kde=True, color='green', ax=ax1)
        ax1.set_title(f'Net Return Long Distribution (%)\n{symbol} {interval}')
        ax1.set_xlabel('Net Return Long (%)')
        ax1.grid(axis='y', linestyle='--')

        short_returns = df_net_returns['Net_Return_Short'].dropna()
        if not short_returns.empty:
            p_low, p_high = short_returns.quantile([0.005, 0.995])
            sns.histplot(data=short_returns[short_returns.between(p_low, p_high)], bins=100, kde=True, color='red', ax=ax2)
        ax2.set_title(f'Net Return Short Distribution (%)\n{symbol} {interval}')
        ax2.set_xlabel('Net Return Short (%)')
        ax2.grid(axis='y', linestyle='--')

        fig.tight_layout()
        return fig

    def plot_label_streak_durations(self, df_streaks: pd.DataFrame, **kwargs) -> 'Figure':
        """Generates histograms of non-zero label streak durations and returns the figure."""
        symbol = kwargs.get('symbol', '')
        interval = kwargs.get('interval', '')
        self.logger.debug(f"Creating Label Streak Duration plot for {symbol} {interval}...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        long_durations = df_streaks.query("label == 1")['duration']
        if not long_durations.empty:
            p_high = long_durations.quantile(0.99)
            sns.histplot(data=long_durations[long_durations < p_high], bins=50, kde=True, color='green', ax=ax1)
        ax1.set_title(f'Long Streak Durations\n{symbol} {interval}')
        ax1.set_xlabel('Duration (bars)')

        short_durations = df_streaks.query("label == -1")['duration']
        if not short_durations.empty:
            p_high = short_durations.quantile(0.99)
            sns.histplot(data=short_durations[short_durations < p_high], bins=50, kde=True, color='red', ax=ax2)
        ax2.set_title(f'Short Streak Durations\n{symbol} {interval}')
        ax2.set_xlabel('Duration (bars)')
        
        fig.tight_layout()
        return fig

    def plot_mfe_mae_distributions(self, df_mfe_mae: pd.DataFrame, **kwargs) -> 'Figure':
        """Generates histograms for MFE and MAE distributions and returns the figure."""
        symbol = kwargs.get('symbol', '')
        interval = kwargs.get('interval', '')
        self.logger.debug(f"Creating MFE/MAE distribution plots for {symbol} {interval}...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        sns.histplot(data=df_mfe_mae.query("label == 1"), x='mfe', bins=50, kde=True, color='green', ax=axes[0, 0])
        axes[0, 0].set_title(f'Long MFE Distribution (%)\n{symbol} {interval}')
        
        sns.histplot(data=df_mfe_mae.query("label == 1"), x='mae', bins=50, kde=True, color='red', ax=axes[0, 1])
        axes[0, 1].set_title(f'Long MAE Distribution (%)\n{symbol} {interval}')

        sns.histplot(data=df_mfe_mae.query("label == -1"), x='mfe', bins=50, kde=True, color='green', ax=axes[1, 0])
        axes[1, 0].set_title(f'Short MFE Distribution (%)\n{symbol} {interval}')

        sns.histplot(data=df_mfe_mae.query("label == -1"), x='mae', bins=50, kde=True, color='red', ax=axes[1, 1])
        axes[1, 1].set_title(f'Short MAE Distribution (%)\n{symbol} {interval}')

        fig.tight_layout()
        return fig

    def plot_mfe_mae_scatter(self, df_mfe_mae: pd.DataFrame, **kwargs) -> 'Figure':
        """Generates a scatter plot of MFE vs. MAE and returns the figure."""
        symbol = kwargs.get('symbol', '')
        interval = kwargs.get('interval', '')
        self.logger.debug(f"Creating MFE vs. MAE scatter plot for {symbol} {interval}...")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=df_mfe_mae, x='mae', y='mfe', hue='label', palette={1: 'green', -1: 'red'}, alpha=0.6, s=20, ax=ax)
        ax.set_title(f'Max Favorable vs. Max Adverse Excursion\n{symbol} {interval}')
        ax.set_xlabel('Max Adverse Excursion (MAE) (%)')
        ax.set_ylabel('Max Favorable Excursion (MFE) (%)')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, ['Short (-1)', 'Long (1)'], title='Label')

        fig.tight_layout()
        return fig

    def plot_future_returns(self, df_returns: pd.DataFrame, **kwargs) -> 'Figure':
        """Generates box plots of future net returns across horizons and returns the figure."""
        symbol = kwargs.get('symbol', '')
        interval = kwargs.get('interval', '')
        self.logger.debug(f"Creating Future Net Returns plot for {symbol} {interval}...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)

        p_low, p_high = df_returns['return_pct'].quantile([0.005, 0.995])
        plot_data = df_returns[df_returns['return_pct'].between(p_low, p_high)]

        sns.boxplot(data=plot_data.query("label == 1"), x='horizon', y='return_pct', hue='horizon', palette='viridis', legend=False, ax=ax1)
        ax1.set_title(f'Long Future Net Return (%)\n{symbol} {interval}')
        ax1.set_ylabel('Net Return (%)')
        ax1.set_xlabel('Horizon (bars)')

        sns.boxplot(data=plot_data.query("label == -1"), x='horizon', y='return_pct', hue='horizon', palette='viridis', legend=False, ax=ax2)
        ax2.set_title(f'Short Future Net Return (%)\n{symbol} {interval}')
        ax2.set_xlabel('Horizon (bars)')

        fig.tight_layout()
        return fig

    def plot_regime_profitability(self, df_regime_returns: pd.DataFrame, **kwargs) -> List[Tuple[str, 'Figure']]:
        """
        Generates a list of box plots for volatility regime net profitability, one per regime.
        """
        symbol = kwargs.get('symbol', '')
        interval = kwargs.get('interval', '')
        self.logger.debug(f"Creating Volatility Regime Profitability plots for {symbol} {interval}...")
        figures = []
        p_low, p_high = df_regime_returns['return_pct'].quantile([0.005, 0.995])
        plot_data_base = df_regime_returns[df_regime_returns['return_pct'].between(p_low, p_high)]

        for regime in sorted(df_regime_returns['volatility_regime'].unique()):
            fig, ax = plt.subplots(figsize=(10, 7))
            plot_data = plot_data_base.query(f"volatility_regime == {regime}")

            if plot_data.empty:
                plt.close(fig)
                continue

            sns.boxplot(data=plot_data, x='horizon', y='return_pct', hue='label', palette={1: 'green', -1: 'red'}, ax=ax)
            ax.set_title(f'Future Net Return Distribution (%) by Label (Regime {regime})\n{symbol} {interval}')
            ax.set_xlabel('Horizon (bars)')
            ax.set_ylabel('Net Return (%)')
            ax.grid(axis='y', linestyle='--')
            
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, ['Short (-1)', 'Long (1)'], title='Label')

            fig.tight_layout()
            analysis_type = f"regime_{regime}_profitability_plot"
            figures.append((analysis_type, fig))
            
        return figures