# utils/labeling/analysis_plotter.py

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__) # Module-level logger for plotting utilities

class AnalysisPlotter:
    """
    Provides utility methods for generating and saving various plots
    related to label analysis.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, plot_pattern: Optional[str] = None):
        """
        Initializes the AnalysisPlotter.

        Args:
            logger (Optional[logging.Logger]): A logger instance for logging messages.
                                               If None, a default logger will be used.
            plot_pattern (Optional[str]): Filename pattern for plots (e.g., "{symbol}_{interval}_{analysis_type}.png").
                                          If None, plotting methods will raise an error if needed.
        """
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.plot_pattern = plot_pattern # Store the pattern as an instance attribute

        self.logger.debug("AnalysisPlotter initialized.")

    def _get_plot_path(self, output_dir: Path, symbol: str, interval: str, analysis_type_suffix: str) -> Path:
        """Helper to construct the full plot file path."""
        if self.plot_pattern is None:
            raise ValueError("plot_pattern is not set in AnalysisPlotter. Cannot generate plot path.")
        
        return output_dir / self.plot_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')

    def plot_label_distribution(self, distribution_summary: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Generates and saves a bar plot of the overall label distribution.

        Args:
            distribution_summary (pd.DataFrame): DataFrame containing 'Label', 'Count', 'Percentage' columns.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            output_dir (Path): Directory to save the plot.
        """
        self.logger.info(f"Generating Label Distribution plot for {symbol.upper()} {interval}...")
        try:
            plt.figure(figsize=(8, 6))
            sns.barplot(x='Label', y='Percentage', data=distribution_summary, palette='viridis', hue='Label', legend=False)
            plt.title(f'Overall Label Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Label (-1: Short, 0: Neutral, 1: Long)')
            plt.ylabel('Percentage (%)')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()

            plot_path = self._get_plot_path(output_dir, symbol, interval, "label_distribution_plot")
            plt.savefig(plot_path, dpi=150)
            self.logger.info(f"Saved label distribution plot to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error plotting label distribution for {symbol} {interval}: {e}", exc_info=True)
        finally:
            plt.close('all')

    def plot_net_return_distributions(self, df_net_returns: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Generates and saves histograms/KDE plots of Net Return Long and Net Return Short distributions.

        Args:
            df_net_returns (pd.DataFrame): DataFrame containing 'Net_Return_Long' and 'Net_Return_Short' columns.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            output_dir (Path): Directory to save the plot.
        """
        self.logger.info(f"Generating Net Return Distributions for {symbol.upper()} {interval}...")
        try:
            plt.figure(figsize=(14, 6))

            long_returns_filtered = df_net_returns['Net_Return_Long'].dropna()
            if not long_returns_filtered.empty:
                long_returns_filtered = long_returns_filtered[long_returns_filtered.between(
                    long_returns_filtered.quantile(0.005), long_returns_filtered.quantile(0.995)
                )]

            short_returns_filtered = df_net_returns['Net_Return_Short'].dropna()
            if not short_returns_filtered.empty:
                short_returns_filtered = short_returns_filtered[short_returns_filtered.between(
                    short_returns_filtered.quantile(0.005), short_returns_filtered.quantile(0.995)
                )]

            plt.subplot(1, 2, 1)
            if not long_returns_filtered.empty:
                sns.histplot(data=long_returns_filtered, bins=100, kde=True, color='green')
                plt.title(f'Net Return Long Distribution (%)\n{symbol.upper()} {interval}')
                plt.xlabel('Net Return Long (%)')
                plt.ylabel('Frequency')
                plt.grid(axis='y', linestyle='--')
            else:
                self.logger.warning("No valid long net returns for plotting.")
                plt.title(f'Net Return Long Distribution (%)\n(No Data)')

            plt.subplot(1, 2, 2)
            if not short_returns_filtered.empty:
                sns.histplot(data=short_returns_filtered, bins=100, kde=True, color='red')
                plt.title(f'Net Return Short Distribution (%)\n{symbol.upper()} {interval}')
                plt.xlabel('Net Return Short (%)')
                plt.ylabel('Frequency')
                plt.grid(axis='y', linestyle='--')
            else:
                self.logger.warning("No valid short net returns for plotting.")
                plt.title(f'Net Return Short Distribution (%)\n(No Data)')

            plt.tight_layout()

            plot_path = self._get_plot_path(output_dir, symbol, interval, "net_return_distributions_plot")
            plt.savefig(plot_path, dpi=150)
            self.logger.info(f"Saved net return distributions plot to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error plotting net return distributions for {symbol} {interval}: {e}", exc_info=True)
        finally:
            plt.close('all')


    def plot_label_streak_durations(self, df_streaks_nonzero: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Generates and saves histograms of non-zero label streak durations.

        Args:
            df_streaks_nonzero (pd.DataFrame): DataFrame containing 'label' and 'duration' for non-zero streaks.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            output_dir (Path): Directory to save the plot.
        """
        self.logger.info(f"Generating Normal Label Streak Duration distribution plots for {symbol.upper()} {interval}...")
        try:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            long_durations = df_streaks_nonzero[df_streaks_nonzero['label'] == 1]['duration']
            if not long_durations.empty:
                long_durations_filtered = long_durations[long_durations < long_durations.quantile(0.99)]
                if not long_durations_filtered.empty:
                    sns.histplot(data=long_durations_filtered, bins=50, kde=True, color='green')
                    plt.title(f'Label 1 (Long) Normal Streak Durations\n{symbol.upper()} {interval}')
                    plt.xlabel('Duration (bars)')
                    plt.ylabel('Frequency')
                    plt.grid(axis='y', linestyle='--')
                else:
                    self.logger.warning("No valid long durations after filtering for plotting.")
                    plt.title(f'Label 1 (Long) Normal Streak Durations\n(No Data)')
            else:
                self.logger.warning("No long streaks to plot.")
                plt.title(f'Label 1 (Long) Normal Streak Durations\n(No Data)')

            plt.subplot(1, 2, 2)
            short_durations = df_streaks_nonzero[df_streaks_nonzero['label'] == -1]['duration']
            if not short_durations.empty:
                short_durations_filtered = short_durations[short_durations < short_durations.quantile(0.99)]
                if not short_durations_filtered.empty:
                    sns.histplot(data=short_durations_filtered, bins=50, kde=True, color='red')
                    plt.title(f'Label -1 (Short) Normal Streak Durations\n{symbol.upper()} {interval}')
                    plt.xlabel('Duration (bars)')
                    plt.ylabel('Frequency')
                    plt.grid(axis='y', linestyle='--')
                else:
                    self.logger.warning("No valid short durations after filtering for plotting.")
                    plt.title(f'Label -1 (Short) Normal Streak Durations\n(No Data)')
            else:
                self.logger.warning("No short streaks to plot.")
                plt.title(f'Label -1 (Short) Normal Streak Durations\n(No Data)')

            plt.tight_layout()

            plot_path = self._get_plot_path(output_dir, symbol, interval, "normal_label_streak_duration_plot")
            plt.savefig(plot_path, dpi=150)
            self.logger.info(f"Saved normal label streak duration plot to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error plotting normal label streak durations for {symbol} {interval}: {e}", exc_info=True)
        finally:
            plt.close('all')

    def plot_mfr_mae_distributions(self, df_max_return_loss: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Generates and saves histograms for Max Favorable Return and Max Adverse Loss distributions.

        Args:
            df_max_return_loss (pd.DataFrame): DataFrame containing 'label', 'max_favorable_return', 'max_adverse_loss'.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            output_dir (Path): Directory to save the plot.
        """
        self.logger.info(f"Generating Max Return/Loss (within f_window) distribution plots for {symbol.upper()} {interval}...")
        try:
            plt.figure(figsize=(12, 12))

            plt.subplot(2, 2, 1)
            sns.histplot(data=df_max_return_loss[df_max_return_loss['label'] == 1], x='max_favorable_return', bins=50, kde=True, color='green')
            plt.title(f'Label 1 (Long) Max Favorable Return (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Max Favorable Return (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(2, 2, 2)
            sns.histplot(data=df_max_return_loss[df_max_return_loss['label'] == 1], x='max_adverse_loss', bins=50, kde=True, color='red')
            plt.title(f'Label 1 (Long) Max Adverse Loss (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Max Adverse Loss (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(2, 2, 3)
            sns.histplot(data=df_max_return_loss[df_max_return_loss['label'] == -1], x='max_favorable_return', bins=50, kde=True, color='green')
            plt.title(f'Label -1 (Short) Max Favorable Return (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Max Favorable Return (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(2, 2, 4)
            sns.histplot(data=df_max_return_loss[df_max_return_loss['label'] == -1], x='max_adverse_loss', bins=50, kde=True, color='red')
            plt.title(f'Label -1 (Short) Max Adverse Loss (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Max Adverse Loss (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.tight_layout()

            plot_path = self._get_plot_path(output_dir, symbol, interval, "max_return_loss_distribution_plot_fwindow")
            plt.savefig(plot_path, dpi=150)
            self.logger.info(f"Saved Max Return/Loss distribution plot to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error plotting Max Return/Loss distributions for {symbol} {interval}: {e}", exc_info=True)
        finally:
            plt.close('all')

    def plot_mfr_mae_scatter(self, df_max_return_loss: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Generates and saves a scatter plot of Max Favorable Return vs. Max Adverse Loss.

        Args:
            df_max_return_loss (pd.DataFrame): DataFrame containing 'label', 'max_favorable_return', 'max_adverse_loss'.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            output_dir (Path): Directory to save the plot.
        """
        self.logger.info(f"Generating MFR vs MAE scatter plot for {symbol.upper()} {interval}...")
        try:
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                data=df_max_return_loss,
                x='max_adverse_loss',
                y='max_favorable_return',
                hue='label',
                palette={1: 'green', -1: 'red'},
                alpha=0.6,
                s=20 # marker size
            )
            plt.title(f'Max Favorable Return vs. Max Adverse Loss\n{symbol.upper()} {interval} (Colored by Label)')
            plt.xlabel('Max Adverse Loss (%)')
            plt.ylabel('Max Favorable Return (%)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.8) # Reference line for 0 MFR
            plt.axvline(0, color='gray', linestyle='--', linewidth=0.8) # Reference line for 0 MAE (though MAE is usually positive)

            handles, labels = plt.gca().get_legend_handles_labels()
            legend_labels_map = {'-1': 'Short (-1)', '1': 'Long (1)'}
            final_handles = []
            final_labels = []
            for label_val in ['-1', '1']:
                if label_val in labels:
                    idx = labels.index(label_val)
                    final_handles.append(handles[idx])
                    final_labels.append(legend_labels_map.get(label_val, label_val))

            if final_handles:
                plt.legend(final_handles, final_labels, title='Label')

            plt.tight_layout()

            plot_path = self._get_plot_path(output_dir, symbol, interval, "mfr_mae_scatter_plot")
            plt.savefig(plot_path, dpi=150)
            self.logger.info(f"Saved MFR vs MAE scatter plot to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error plotting MFR vs MAE scatter plot for {symbol} {interval}: {e}", exc_info=True)
        finally:
            plt.close('all')

    def plot_future_returns(self, df_returns: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Generates and saves box plots of future net returns for long and short labels across horizons.

        Args:
            df_returns (pd.DataFrame): DataFrame containing 'label', 'horizon', 'return_pct'.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            output_dir (Path): Directory to save the plot.
        """
        self.logger.info(f"Generating Future Net Returns distribution plots for {symbol.upper()} {interval}...")
        try:
            plt.figure(figsize=(14, 8))

            plt.subplot(1, 2, 1)
            sns.boxplot(data=df_returns[(df_returns['label'] == 1) & (df_returns['return_pct'].between(df_returns['return_pct'].quantile(0.005), df_returns['return_pct'].quantile(0.995)))],
                        x='horizon', y='return_pct', hue='horizon', palette='viridis', legend=False)
            plt.title(f'Label 1 (Long) Future Net Return Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Horizon (bars)')
            plt.ylabel('Net Return (%)')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(1, 2, 2)
            sns.boxplot(data=df_returns[(df_returns['label'] == -1) & (df_returns['return_pct'].between(df_returns['return_pct'].quantile(0.005), df_returns['return_pct'].quantile(0.995)))],
                        x='horizon', y='return_pct', hue='horizon', palette='viridis', legend=False)
            plt.title(f'Label -1 (Short) Future Net Return Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Horizon (bars)')
            plt.ylabel('Net Return (%)')
            plt.grid(axis='y', linestyle='--')

            plt.tight_layout()

            plot_path = self._get_plot_path(output_dir, symbol, interval, "future_net_returns_distribution_plot")
            plt.savefig(plot_path, dpi=150)
            self.logger.info(f"Saved future net returns distribution plot to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error plotting future returns for {symbol} {interval}: {e}", exc_info=True)
        finally:
            plt.close('all')

    def plot_regime_profitability(self, df_regime_returns: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Generates and saves box plots of volatility regime net profitability.

        Args:
            df_regime_returns (pd.DataFrame): DataFrame containing 'label', 'volatility_regime', 'horizon', 'return_pct'.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            output_dir (Path): Directory to save the plot.
        """
        self.logger.info(f"Generating Volatility Regime Net Profitability plots for {symbol.upper()} {interval}...")
        try:
            for regime in sorted(df_regime_returns['volatility_regime'].unique()):
                plt.figure(figsize=(14, 8))
                plot_data = df_regime_returns[(df_regime_returns['volatility_regime'] == regime) &
                                              (df_regime_returns['return_pct'].between(df_regime_returns['return_pct'].quantile(0.005), df_regime_returns['return_pct'].quantile(0.995)))].copy()

                if plot_data.empty:
                    self.logger.warning(f"No data for plotting regime profitability for regime {regime}. Skipping plot.")
                    plt.close()
                    continue

                ax = sns.boxplot(data=plot_data, x='horizon', y='return_pct', hue='label',
                                 palette={1: 'green', -1: 'red'})

                plt.title(f'Mean Future Net Return (%) by Label (Regime {regime})\n{symbol.upper()} {interval}')
                plt.xlabel('Horizon (bars)')
                plt.ylabel('Net Return (%)')
                plt.grid(axis='y', linestyle='--')

                handles, labels = ax.get_legend_handles_labels()
                legend_labels_map = {'-1': 'Short (-1)', '1': 'Long (1)'}

                final_handles = []
                final_labels = []
                for label_val in [-1, 1]:
                    label_str = str(label_val)
                    if label_str in labels:
                        idx = labels.index(label_str)
                        final_handles.append(handles[idx])
                        final_labels.append(legend_labels_map.get(label_str, label_str))

                if final_handles:
                    plt.legend(final_handles, final_labels, title='Label')

                plt.tight_layout()

                plot_path = self._get_plot_path(output_dir, symbol, interval, f"regime_{regime}_net_profitability_plot")
                plt.savefig(plot_path, dpi=150)
                self.logger.info(f"Saved volatility regime profitability plot (Regime {regime}) to {plot_path}")
        except Exception as e:
            self.logger.error(f"Error plotting volatility regime profitability for {symbol} {interval}: {e}", exc_info=True)
        finally:
            plt.close('all')

