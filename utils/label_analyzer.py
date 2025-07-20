# utils/label_analyzer.py

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Tuple, List

# Set up a module-level logger
logger = logging.getLogger(__name__)

# Define FLOAT_EPSILON for robust floating-point comparisons
FLOAT_EPSILON = 1e-9

class LabelAnalyzer:
    """
    Performs various analyses on generated trading labels and processed OHLCV data.
    Results (plots and tables) are saved to strategy-specific subfolders.
    """

    def __init__(self, paths: Dict[str, Any], logger: logging.Logger, fee: float, slippage: float, f_window: int):
        """
        Initializes the LabelAnalyzer.

        Args:
            paths (Dict[str, Any]): The PATHS dictionary from config/paths.py.
            logger (logging.Logger): A logger instance for logging messages.
            fee (float): Transaction fee rate (e.g., 0.0005).
            slippage (float): Slippage rate (e.g., 0.0001).
            f_window (int): The forward lookahead window used by the labeling strategy.
                            This will be used as the maximum holding period for MFE/MAE and streak analysis.
        """
        self.paths = paths
        self.logger = logger
        self.fee = fee
        self.slippage = slippage
        self.f_window = f_window # Store f_window here
        self.logger.info(f"LabelAnalyzer initialized with fee={self.fee}, slippage={self.slippage}, and f_window={self.f_window}.")

        # Ensure required patterns are available
        self.plot_pattern = self.paths.get("labeling_analysis_plot_pattern")
        self.table_pattern = self.paths.get("labeling_analysis_table_pattern")
        self.strategy_dir_pattern_str = self.paths.get("labeling_strategy_analysis_dir_pattern")

        if not all([self.plot_pattern, self.table_pattern, self.strategy_dir_pattern_str]):
            self.logger.error("Missing required analysis patterns in PATHS. Check config/paths.py.")
            raise ValueError("Missing required analysis patterns in PATHS.")

    def _calculate_net_return_scalar(self, entry_price: float, exit_price: float, trade_type: int) -> float:
        """
        Calculates net return for scalar values, accounting for fees and slippage.
        Returns percentage.

        Args:
            entry_price (float): The price at which the trade is considered entered.
            exit_price (float): The price at which the trade is considered exited.
            trade_type (int): 1 for long, -1 for short.

        Returns:
            float: Net return percentage.
        """
        if abs(entry_price) < FLOAT_EPSILON:
            return np.nan # Avoid division by zero

        if trade_type == 1: # Long trade: Buy at entry, Sell at exit
            # Cost to Enter (Buy): entry_price * (1 + fee + slippage)
            # Revenue from Exit (Sell): exit_price * (1 - fee - slippage)
            cost_to_enter = entry_price * (1 + self.fee + self.slippage)
            revenue_from_exit = exit_price * (1 - self.fee - self.slippage)
            net_return = ((revenue_from_exit - cost_to_enter) / cost_to_enter) * 100.0
        elif trade_type == -1: # Short trade: Sell at entry, Buy at exit
            # Revenue from Enter (Sell): entry_price * (1 - fee - slippage)
            # Cost to Exit (Buy Back): exit_price * (1 + fee + slippage)
            revenue_from_enter = entry_price * (1 - self.fee - self.slippage)
            cost_to_exit = exit_price * (1 + self.fee + self.slippage)
            net_return = ((revenue_from_enter - cost_to_exit) / revenue_from_enter) * 100.0
        else:
            net_return = np.nan # Should not happen for labels 1 or -1

        return net_return

    def _calculate_max_return_loss_within_fwindow(self, df_segment: pd.DataFrame, entry_price: float, trade_type: int) -> Tuple[float, float]:
        """
        Calculates Maximum Favorable Return and Maximum Adverse Loss for a given price segment
        (representing the f_window duration) starting from an entry price, accounting for fees and slippage.

        Args:
            df_segment (pd.DataFrame): DataFrame slice containing OHLC data for the trade duration (up to f_window).
                                       Must have 'high' and 'low' columns.
            entry_price (float): The price at which the trade is considered entered.
            trade_type (int): 1 for long, -1 for short.

        Returns:
            Tuple[float, float]: (Max Favorable Return percentage, Max Adverse Loss percentage).
                                 Returns (np.nan, np.nan) if segment is empty.
        """
        if df_segment.empty:
            return np.nan, np.nan

        df_segment = df_segment.copy() # Work on a copy to avoid SettingWithCopyWarning

        # Ensure high and low columns are numeric, coercing errors to NaN
        df_segment['high'] = pd.to_numeric(df_segment['high'], errors='coerce')
        df_segment['low'] = pd.to_numeric(df_segment['low'], errors='coerce')

        # Drop rows where high/low became NaN after coercion
        df_segment.dropna(subset=['high', 'low'], inplace=True)

        if df_segment.empty:
             return np.nan, np.nan # Return NaN if segment becomes empty after dropping NaNs

        max_favorable_return_pct = np.nan
        max_adverse_loss_pct = np.nan

        # Initialize with values that will be easily overwritten by any actual movement
        # For max favorable, start at negative infinity
        # For max adverse, start at negative infinity (since we take abs later)
        current_max_favorable = -np.inf
        current_max_adverse = -np.inf


        if trade_type == 1: # Long trade
            for _, row in df_segment.iterrows():
                # Favorable: price goes up (use high)
                favorable_return = self._calculate_net_return_scalar(entry_price, row['high'], trade_type=1)
                if pd.notna(favorable_return):
                    current_max_favorable = max(current_max_favorable, favorable_return)

                # Adverse: price goes down (use low)
                adverse_return = self._calculate_net_return_scalar(entry_price, row['low'], trade_type=1)
                if pd.notna(adverse_return):
                    current_max_adverse = max(current_max_adverse, abs(adverse_return)) # MAE is positive magnitude

        elif trade_type == -1: # Short trade
            for _, row in df_segment.iterrows():
                # Favorable: price goes down (use low)
                favorable_return = self._calculate_net_return_scalar(entry_price, row['low'], trade_type=-1)
                if pd.notna(favorable_return):
                    current_max_favorable = max(current_max_favorable, favorable_return)

                # Adverse: price goes up (use high)
                adverse_return = self._calculate_net_return_scalar(entry_price, row['high'], trade_type=-1)
                if pd.notna(adverse_return):
                    current_max_adverse = max(current_max_adverse, abs(adverse_return)) # MAE is positive magnitude

        # If no valid movements were found, keep as NaN
        max_favorable_return_pct = current_max_favorable if current_max_favorable != -np.inf else np.nan
        max_adverse_loss_pct = current_max_adverse if current_max_adverse != -np.inf else np.nan

        return max_favorable_return_pct, max_adverse_loss_pct

    def analyze_label_distribution(self, df_labeled: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Analyzes the overall distribution (counts and percentages) of labels (-1, 0, 1).
        """
        self.logger.info(f"Starting Label Distribution Analysis for {symbol.upper()} {interval}...")

        if df_labeled.empty or 'label' not in df_labeled.columns:
            self.logger.warning("Labeled DataFrame is empty or missing 'label' column. Skipping label distribution analysis.")
            return

        df_labeled = df_labeled.copy()
        df_labeled['label'] = pd.to_numeric(df_labeled['label'], errors='coerce').fillna(0).astype(int)

        label_counts = df_labeled['label'].value_counts().sort_index()
        total_labels = label_counts.sum()
        label_percentages = (label_counts / total_labels) * 100 if total_labels > 0 else pd.Series(dtype=float)

        distribution_summary = pd.DataFrame({
            'Label': label_counts.index,
            'Count': label_counts.values,
            'Percentage': label_percentages.values
        })

        self.logger.info("\nOverall Label Distribution:")
        self.logger.info(distribution_summary.to_string(index=False))

        analysis_type_suffix = "label_distribution"
        table_path = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')
        try:
            distribution_summary.to_csv(table_path, index=False)
            self.logger.info(f"Saved label distribution summary to {table_path}")
        except Exception as e:
            self.logger.error(f"Failed to save label distribution summary table: {e}")

        # Optional: Plotting the distribution
        try:
            plt.figure(figsize=(8, 6))
            sns.barplot(x='Label', y='Percentage', data=distribution_summary, palette='viridis')
            plt.title(f'Overall Label Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Label (-1: Short, 0: Neutral, 1: Long)')
            plt.ylabel('Percentage (%)')
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()

            plot_type_suffix = "label_distribution_plot"
            plot_path = output_dir / self.plot_pattern.format(
                symbol=symbol.upper(), interval=interval, analysis_type=plot_type_suffix
            ).replace(':', '_')
            try:
                plt.savefig(plot_path, dpi=150)
                self.logger.info(f"Saved label distribution plot to {plot_path}")
            except Exception as e:
                self.logger.error(f"Failed to save label distribution plot: {e}")
        except ImportError:
            self.logger.warning("Matplotlib or Seaborn not installed. Skipping plot generation for label distribution.")
        except Exception as e:
            self.logger.error(f"Error plotting label distribution: {e}", exc_info=True)
        finally:
            if 'plt' in locals() and plt.get_fignums():
                 plt.close('all')

        self.logger.info("Label Distribution Analysis complete.")


    def analyze_label_streaks(self, df_labeled: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Analyzes the duration of consecutive non-zero label streaks,
        where 0 means 'hold' the previous active position,
        and the streak is capped by the f_window of the initial signal.
        """
        self.logger.info(f"Starting Label Streak Analysis for {symbol.upper()} {interval} (0 means hold, capped by f_window)...")

        if df_labeled.empty or 'label' not in df_labeled.columns:
            self.logger.warning("Labeled DataFrame is empty or missing 'label' column. Skipping streak analysis.")
            return

        df_labeled = df_labeled.copy() # Work on a copy
        # Ensure label is integer type
        df_labeled['label'] = pd.to_numeric(df_labeled['label'], errors='coerce').fillna(0).astype(int)

        streak_data = []
        current_active_label = 0 # 0: no active position, 1: long, -1: short
        current_start_index_iloc = None # Use iloc for easier arithmetic

        n = len(df_labeled)
        for i in range(n):
            label = df_labeled['label'].iloc[i]

            if current_active_label == 0: # No active position
                if label != 0: # New active signal
                    current_active_label = label
                    current_start_index_iloc = i
            else: # An active position (1 or -1) is held
                # Check if label flips to opposite signal OR f_window is reached
                if label == -current_active_label or (i - current_start_index_iloc >= self.f_window):
                    streak_length = i - current_start_index_iloc
                    if streak_length > 0: # Ensure at least 1 bar duration
                        streak_data.append({'label': current_active_label, 'duration': streak_length})

                    # Reset for new streak or no position
                    if label == -current_active_label: # Flipped, so start new streak with this label
                        current_active_label = label
                        current_start_index_iloc = i
                    else: # f_window reached, and current label is 0 or same, so end position
                        current_active_label = 0
                        current_start_index_iloc = None
                # If label is 0 or same as current_active_label, and f_window not reached, continue holding, streak continues
                # No action needed, current_active_label and current_start_index_iloc remain

        # Handle the last streak if it extends to the end of the DataFrame
        if current_active_label != 0 and current_start_index_iloc is not None:
            streak_length = n - current_start_index_iloc
            if streak_length > 0:
                streak_data.append({'label': current_active_label, 'duration': streak_length})


        if not streak_data:
            self.logger.info("No active position streaks found for analysis.")
            return

        df_streaks = pd.DataFrame(streak_data)
        df_streaks_nonzero = df_streaks[df_streaks['label'] != 0].copy()

        if df_streaks_nonzero.empty:
            self.logger.info("No non-zero active position streaks found for duration analysis after filtering.")
            return

        self.logger.info("Calculating streak duration summary statistics...")
        if len(df_streaks_nonzero) > 0:
            agg_funcs = ['count', 'mean', 'median', 'std']
            # Dynamically add quantiles only if enough data points exist
            quantiles_to_add = [0.75, 0.90, 0.95]
            for q in quantiles_to_add:
                # Check if there are enough non-NaN values to compute the quantile
                if df_streaks_nonzero['duration'].count() >= (1 / (1 - q)):
                    agg_funcs.append(lambda x, q=q: x.quantile(q))
                else:
                    self.logger.warning(f"Not enough data points ({df_streaks_nonzero['duration'].count()}) for {q*100}th percentile calculation. Skipping.")

            streak_summary = df_streaks_nonzero.groupby('label')['duration'].agg(agg_funcs).reset_index()

            # Construct column names based on which quantiles were actually added
            col_names = ['Label', 'Count', 'Mean Duration', 'Median Duration', 'Std Dev Duration']
            if any(q == 0.75 for q in quantiles_to_add if df_streaks_nonzero['duration'].count() >= (1 / (1 - q))):
                col_names.append('75th Percentile')
            if any(q == 0.90 for q in quantiles_to_add if df_streaks_nonzero['duration'].count() >= (1 / (1 - q))):
                col_names.append('90th Percentile')
            if any(q == 0.95 for q in quantiles_to_add if df_streaks_nonzero['duration'].count() >= (1 / (1 - q))):
                col_names.append('95th Percentile')

            streak_summary.columns = col_names
        else:
            self.logger.warning("No non-zero streaks to calculate summary statistics.")
            streak_summary = pd.DataFrame(columns=['Label', 'Count', 'Mean Duration', 'Median Duration', 'Std Dev Duration'])


        self.logger.info("\nActive Position Streak Duration Summary (in bars, capped by f_window):") # Changed title
        self.logger.info(streak_summary.to_string())

        analysis_type_suffix = "active_position_streak_duration_fwindow_capped" # Changed suffix
        table_path = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')
        try:
            streak_summary.to_csv(table_path, index=False)
            self.logger.info(f"Saved active position streak duration summary to {table_path}")
        except Exception as e:
            self.logger.error(f"Failed to save active position streak duration summary table: {e}")

        self.logger.info("Generating active position streak duration distribution plots (capped by f_window)...") # Changed title
        try:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            # Filter for label 1 and exclude extreme outliers for better visualization
            long_durations = df_streaks_nonzero[(df_streaks_nonzero['label'] == 1)]['duration']
            if not long_durations.empty:
                long_durations_filtered = long_durations[long_durations < long_durations.quantile(0.99)]
                if not long_durations_filtered.empty:
                    sns.histplot(data=long_durations_filtered, bins=50, kde=True, color='green')
                    plt.title(f'Label 1 (Long) Active Position Streak Durations (0=Hold, f_window capped)\n{symbol.upper()} {interval}')
                    plt.xlabel('Duration (bars)')
                    plt.ylabel('Frequency')
                    plt.grid(axis='y', linestyle='--')
                else:
                    self.logger.warning("No valid long durations after filtering for plotting.")
                    plt.title(f'Label 1 (Long) Active Position Streak Durations\n(No Data)')
            else:
                self.logger.warning("No long active position streaks to plot.")
                plt.title(f'Label 1 (Long) Active Position Streak Durations\n(No Data)')


            plt.subplot(1, 2, 2)
            # Filter for label -1 and exclude extreme outliers for better visualization
            short_durations = df_streaks_nonzero[(df_streaks_nonzero['label'] == -1)]['duration']
            if not short_durations.empty:
                short_durations_filtered = short_durations[short_durations < short_durations.quantile(0.99)]
                if not short_durations_filtered.empty:
                    sns.histplot(data=short_durations_filtered, bins=50, kde=True, color='red')
                    plt.title(f'Label -1 (Short) Active Position Streak Durations (0=Hold, f_window capped)\n{symbol.upper()} {interval}')
                    plt.xlabel('Duration (bars)')
                    plt.ylabel('Frequency')
                    plt.grid(axis='y', linestyle='--')
                else:
                    self.logger.warning("No valid short durations after filtering for plotting.")
                    plt.title(f'Label -1 (Short) Active Position Streak Durations\n(No Data)')
            else:
                self.logger.warning("No short active position streaks to plot.")
                plt.title(f'Label -1 (Short) Active Position Streak Durations\n(No Data)')


            plt.tight_layout()

            plot_type_suffix = "active_position_streak_duration_plot_fwindow_capped" # Changed suffix
            plot_path = output_dir / self.plot_pattern.format(
                symbol=symbol.upper(), interval=interval, analysis_type=plot_type_suffix
            ).replace(':', '_')
            try:
                plt.savefig(plot_path, dpi=150)
                self.logger.info(f"Saved active position streak duration plot to {plot_path}")
            except Exception as e:
                self.logger.error(f"Failed to save active position streak duration plot: {e}")

        except ImportError:
            self.logger.warning("Matplotlib or Seaborn not installed. Skipping plot generation.")
        except Exception as e:
            self.logger.error(f"Error plotting active position streak durations: {e}", exc_info=True)
        finally:
            if 'plt' in locals() and plt.get_fignums():
                 plt.close('all')

        self.logger.info("Active Position Streak Analysis (f_window capped) complete.") # Changed title


    def analyze_mfe_mae(self, df_combined: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
        """
        Analyzes Maximum Favorable Return and Maximum Adverse Loss for periods following
        non-zero labels, calculated over the f_window of the initial signal.

        Assumes df_combined contains 'open', 'high', 'low', 'close' and 'label' columns
        and is indexed by time.
        """
        self.logger.info(f"Starting Max Return/Loss within f_window Analysis for {symbol.upper()} {interval}...")

        required_cols = ['open', 'high', 'low', 'close', 'label']
        if not all(col in df_combined.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_combined.columns]
            self.logger.error(f"Missing required columns for Max Return/Loss analysis: {missing}. Skipping.")
            return

        df_combined = df_combined.copy() # Work on a copy
        # Ensure OHLC and label columns are numeric, coercing errors to NaN
        for col in ['open', 'high', 'low', 'close', 'label']:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

        # Drop rows where critical OHLC or label became NaN
        df_combined.dropna(subset=['open', 'high', 'low', 'close', 'label'], inplace=True)

        if df_combined.empty:
            self.logger.warning("DataFrame is empty after dropping NaNs in critical columns. Skipping Max Return/Loss analysis.")
            return

        # Ensure label is integer type after dropping NaNs
        df_combined['label'] = df_combined['label'].astype(int)


        max_return_loss_data = []
        n = len(df_combined)

        # Iterate through the DataFrame to find entry points for active signals
        for i in range(n):
            current_label = df_combined['label'].iloc[i]

            if current_label != 0: # Only analyze if there's an initial non-neutral signal
                entry_price_ref = df_combined['close'].iloc[i]

                # Define the segment for analysis: from the next bar up to f_window bars later
                segment_end_iloc = min(i + self.f_window + 1, n) # +1 because iloc is exclusive end
                price_segment_df = df_combined.iloc[i + 1 : segment_end_iloc].copy()

                if price_segment_df.empty:
                    self.logger.warning(f"No valid price segment for analysis for signal at index {df_combined.index[i]}. Skipping.")
                    continue

                max_favorable, max_adverse = self._calculate_max_return_loss_within_fwindow(
                    price_segment_df, entry_price_ref, current_label
                )
                max_return_loss_data.append({
                    'label': current_label,
                    'max_favorable_return': max_favorable,
                    'max_adverse_loss': max_adverse
                })


        if not max_return_loss_data:
            self.logger.info("No active signals found for Max Return/Loss analysis.")
            return None # Return None if no data, so ratio analysis can skip

        df_max_return_loss = pd.DataFrame(max_return_loss_data)
        df_max_return_loss.dropna(subset=['max_favorable_return', 'max_adverse_loss'], inplace=True)

        if df_max_return_loss.empty:
            self.logger.info("Max Return/Loss DataFrame is empty after dropping NaNs.")
            return None # Return None if empty after dropping NaNs

        self.logger.info("Calculating Max Return/Loss summary statistics...")
        if len(df_max_return_loss) > 0:
            mfr_agg_funcs = ['count', 'mean', 'median']
            mal_agg_funcs = ['count', 'mean', 'median']
            quantiles_to_add = [0.75, 0.90, 0.95]
            for q in quantiles_to_add:
                if df_max_return_loss['max_favorable_return'].count() >= (1 / (1 - q)):
                     mfr_agg_funcs.append(lambda x, q=q: x.quantile(q))
                     mal_agg_funcs.append(lambda x, q=q: x.quantile(q))
                else:
                     self.logger.warning(f"Not enough data points ({df_max_return_loss['max_favorable_return'].count()}) for {q*100}th percentile calculation. Skipping.")


            mfr_summary = df_max_return_loss.groupby('label')['max_favorable_return'].agg(mfr_agg_funcs).reset_index()
            mal_summary = df_max_return_loss.groupby('label')['max_adverse_loss'].agg(mal_agg_funcs).reset_index()

            mfr_col_names = ['Label', 'Count', 'Mean Max Favorable Return', 'Median Max Favorable Return']
            mal_col_names = ['Label', 'Count', 'Mean Max Adverse Loss', 'Median Max Adverse Loss']

            if any(q == 0.75 for q in quantiles_to_add if df_max_return_loss['max_favorable_return'].count() >= (1 / (1 - q))):
                mfr_col_names.append('75th Percentile Max Favorable Return')
                mal_col_names.append('75th Percentile Max Adverse Loss')
            if any(q == 0.90 for q in quantiles_to_add if df_max_return_loss['max_favorable_return'].count() >= (1 / (1 - q))):
                mfr_col_names.append('90th Percentile Max Favorable Return')
                mal_col_names.append('90th Percentile Max Adverse Loss')
            if any(q == 0.95 for q in quantiles_to_add if df_max_return_loss['max_favorable_return'].count() >= (1 / (1 - q))):
                mfr_col_names.append('95th Percentile Max Favorable Return')
                mal_col_names.append('95th Percentile Max Adverse Loss')

            mfr_summary.columns = mfr_col_names
            mal_summary.columns = mal_col_names
        else:
            self.logger.warning("No Max Return/Loss data to calculate summary statistics.")
            mfr_summary = pd.DataFrame(columns=['Label', 'Count', 'Mean Max Favorable Return', 'Median Max Favorable Return'])
            mal_summary = pd.DataFrame(columns=['Label', 'Count', 'Mean Max Adverse Loss', 'Median Max Adverse Loss'])


        self.logger.info("\nMaximum Favorable Return (within f_window) Summary (%):")
        self.logger.info(mfr_summary.to_string())

        self.logger.info("\nMaximum Adverse Loss (within f_window) Summary (%):")
        self.logger.info(mal_summary.to_string())

        analysis_type_suffix_mfr = "max_favorable_return_fwindow" # Changed suffix
        table_path_mfr = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix_mfr
        ).replace(':', '_')
        try:
            mfr_summary.to_csv(table_path_mfr, index=False)
            self.logger.info(f"Saved Max Favorable Return summary table to {table_path_mfr}")
        except Exception as e:
            self.logger.error(f"Failed to save Max Favorable Return summary table: {e}")

        analysis_type_suffix_mal = "max_adverse_loss_fwindow" # Changed suffix
        table_path_mal = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix_mal
        ).replace(':', '_')
        try:
            mal_summary.to_csv(table_path_mal, index=False)
            self.logger.info(f"Saved Max Adverse Loss summary table to {table_path_mal}")
        except Exception as e:
            self.logger.error(f"Failed to save Max Adverse Loss summary table: {e}")


        self.logger.info("Generating Max Return/Loss (within f_window) distribution plots...") # Changed title
        try:
            plt.figure(figsize=(12, 12))

            plt.subplot(2, 2, 1)
            sns.histplot(data=df_max_return_loss[df_max_return_loss['label'] == 1], x='max_favorable_return', bins=50, kde=True, color='green')
            plt.title(f'Label 1 (Long) Max Favorable Return (%) (within f_window)\n{symbol.upper()} {interval}') # Changed title
            plt.xlabel('Max Favorable Return (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(2, 2, 2)
            sns.histplot(data=df_max_return_loss[df_max_return_loss['label'] == 1], x='max_adverse_loss', bins=50, kde=True, color='red')
            plt.title(f'Label 1 (Long) Max Adverse Loss (%) (within f_window)\n{symbol.upper()} {interval}') # Changed title
            plt.xlabel('Max Adverse Loss (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(2, 2, 3)
            sns.histplot(data=df_max_return_loss[df_max_return_loss['label'] == -1], x='max_favorable_return', bins=50, kde=True, color='green')
            plt.title(f'Label -1 (Short) Max Favorable Return (%) (within f_window)\n{symbol.upper()} {interval}') # Changed title
            plt.xlabel('Max Favorable Return (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(2, 2, 4)
            sns.histplot(data=df_max_return_loss[df_max_return_loss['label'] == -1], x='max_adverse_loss', bins=50, kde=True, color='red')
            plt.title(f'Label -1 (Short) Max Adverse Loss (%) (within f_window)\n{symbol.upper()} {interval}') # Changed title
            plt.xlabel('Max Adverse Loss (%)')
            plt.ylabel('Frequency')
            plt.grid(axis='y', linestyle='--')

            plt.tight_layout()

            plot_type_suffix = "max_return_loss_distribution_plot_fwindow" # Changed suffix
            plot_path = output_dir / self.plot_pattern.format(
                symbol=symbol.upper(), interval=interval, analysis_type=plot_type_suffix
            ).replace(':', '_')
            try:
                plt.savefig(plot_path, dpi=150)
                self.logger.info(f"Saved Max Return/Loss distribution plot to {plot_path}")
            except Exception as e:
                self.logger.error(f"Failed to save Max Return/Loss distribution plot: {e}")

            # --- New Scatter Plot for MFR vs MAE ---
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

            # Manually create legend handles and labels for clarity
            handles, labels = plt.gca().get_legend_handles_labels()
            legend_labels_map = {'-1': 'Short (-1)', '1': 'Long (1)'}
            final_handles = []
            final_labels = []
            for label_val in ['-1', '1']: # Iterate through expected labels in a desired order
                if label_val in labels:
                    idx = labels.index(label_val)
                    final_handles.append(handles[idx])
                    final_labels.append(legend_labels_map.get(label_val, label_val))

            if final_handles:
                plt.legend(final_handles, final_labels, title='Label')

            plt.tight_layout()

            plot_type_suffix_scatter = "mfr_mae_scatter_plot"
            plot_path_scatter = output_dir / self.plot_pattern.format(
                symbol=symbol.upper(), interval=interval, analysis_type=plot_type_suffix_scatter
            ).replace(':', '_')
            try:
                plt.savefig(plot_path_scatter, dpi=150)
                self.logger.info(f"Saved MFR vs MAE scatter plot to {plot_path_scatter}")
            except Exception as e:
                self.logger.error(f"Failed to save MFR vs MAE scatter plot: {e}")

        except ImportError:
            self.logger.warning("Matplotlib or Seaborn not installed. Skipping plot generation.")
        except Exception as e:
            self.logger.error(f"Error plotting Max Return/Loss distributions or scatter plot: {e}", exc_info=True)
        finally:
            if 'plt' in locals() and plt.get_fignums():
                 plt.close('all')

        self.logger.info("Max Return/Loss (within f_window) Analysis complete.")
        return df_max_return_loss # Return the DataFrame for ratio analysis


    # The analyze_max_return_loss_ratio function has been removed as requested.
    # def analyze_max_return_loss_ratio(self, df_max_return_loss: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
    #     """
    #     Analyzes the distribution of the (Max Favorable Return / Max Adverse Loss) ratio for each label.
    #     ... (rest of the function)
    #     """
    #     pass # This function is now removed


    def analyze_future_returns(self, df_combined: pd.DataFrame, symbol: str, interval: str, output_dir: Path, horizons: List[int]):
        """
        Analyzes future NET percentage returns over specified horizons following non-zero labels.

        Assumes df_combined contains 'close' and 'label' columns and is indexed by time.
        """
        self.logger.info(f"Starting Future NET Returns Analysis for {symbol.upper()} {interval} over horizons {horizons} bars...")

        required_cols = ['close', 'label']
        if not all(col in df_combined.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_combined.columns]
            self.logger.error(f"Missing required columns for Future Returns analysis: {missing}. Skipping.")
            return

        df_combined = df_combined.copy() # Work on a copy
        # Ensure close and label columns are numeric, coercing errors to NaN
        for col in ['close', 'label']:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

        # Drop rows where critical close or label became NaN
        df_combined.dropna(subset=['close', 'label'], inplace=True)

        if df_combined.empty:
            self.logger.warning("DataFrame is empty after dropping NaNs in critical columns. Skipping Future Returns analysis.")
            return

        # Ensure label is integer type after dropping NaNs
        df_combined['label'] = df_combined['label'].astype(int)


        returns_data = []

        # The logic for Future Returns analysis (fixed horizon) remains the same.
        # It's about the potential return if you entered at a signal and held for exactly 'horizon' bars.
        # The '0' as hold logic within the f_window is not directly relevant here,
        # as this analysis is about the *potential* return at a fixed future point,
        # not the actual trade management.
        for horizon in horizons:
            future_close_series = df_combined['close'].shift(-horizon)

            for i in range(len(df_combined)):
                current_label = df_combined['label'].iloc[i]
                current_close = df_combined['close'].iloc[i]
                future_close = future_close_series.iloc[i]

                # Only consider points where a non-neutral signal was given as an "entry" for this analysis
                if current_label != 0 and pd.notna(current_close) and pd.notna(future_close) and abs(current_close) > FLOAT_EPSILON:
                    net_return = self._calculate_net_return_scalar(current_close, future_close, current_label) # Use 'current_label' as trade_type
                    if pd.notna(net_return):
                        returns_data.append({'label': current_label, 'horizon': horizon, 'return_pct': net_return})


        if not returns_data:
            self.logger.info("No non-zero labels with valid future net returns found for analysis.")
            return

        df_returns = pd.DataFrame(returns_data)

        self.logger.info("Calculating Future Net Returns summary statistics...")
        if len(df_returns) > 0:
            agg_funcs = ['count', 'mean', 'median', 'std', 'min', 'max']
            agg_funcs_with_quantiles = agg_funcs + [lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]

            returns_summary = df_returns.groupby(['label', 'horizon'])['return_pct'].agg(agg_funcs_with_quantiles).reset_index()
            returns_summary.columns = ['Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile']

        else:
            self.logger.warning("No future returns data to calculate summary statistics.")
            returns_summary = pd.DataFrame(columns=['Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile'])


        self.logger.info("\nFuture Net Returns Summary (%) by Label and Horizon:")
        self.logger.info(returns_summary.to_string())

        analysis_type_suffix = "future_net_returns_summary"
        table_path = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')
        try:
            returns_summary.to_csv(table_path, index=False)
            self.logger.info(f"Saved future net returns summary to {table_path}")
        except Exception as e:
            self.logger.error(f"Failed to save future net returns summary table: {e}")

        self.logger.info("Generating Future Net Returns distribution plots...")
        try:
            plt.figure(figsize=(14, 8))

            plt.subplot(1, 2, 1)
            # Use hue='horizon' for color variation within the subplot, no legend for label here
            sns.boxplot(data=df_returns[(df_returns['label'] == 1) & (df_returns['return_pct'].between(df_returns['return_pct'].quantile(0.005), df_returns['return_pct'].quantile(0.995)))],
                        x='horizon', y='return_pct', hue='horizon', palette='viridis', legend=False)
            plt.title(f'Label 1 (Long) Future Net Return Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Horizon (bars)')
            plt.ylabel('Net Return (%)')
            plt.grid(axis='y', linestyle='--')

            plt.subplot(1, 2, 2)
            # Use hue='horizon' for color variation within the subplot, no legend for label here
            sns.boxplot(data=df_returns[(df_returns['label'] == -1) & (df_returns['return_pct'].between(df_returns['return_pct'].quantile(0.005), df_returns['return_pct'].quantile(0.995)))],
                        x='horizon', y='return_pct', hue='horizon', palette='viridis', legend=False)
            plt.title(f'Label -1 (Short) Future Net Return Distribution (%)\n{symbol.upper()} {interval}')
            plt.xlabel('Horizon (bars)')
            plt.ylabel('Net Return (%)')
            plt.grid(axis='y', linestyle='--')

            plt.tight_layout()

            plot_type_suffix = "future_net_returns_distribution_plot"
            plot_path = output_dir / self.plot_pattern.format(
                symbol=symbol.upper(), interval=interval, analysis_type=plot_type_suffix
            ).replace(':', '_')
            try:
                plt.savefig(plot_path, dpi=150)
                self.logger.info(f"Saved future net returns distribution plot to {plot_path}")
            except Exception as e:
                self.logger.error(f"Failed to save future net returns distribution plot: {e}")

        except ImportError:
            self.logger.warning("Matplotlib or Seaborn not installed. Skipping plot generation.")
        except Exception as e:
            self.logger.error(f"Error plotting future returns: {e}", exc_info=True)
        finally:
            if 'plt' in locals() and plt.get_fignums():
                 plt.close('all')

        self.logger.info("Future Net Returns Analysis complete.")


    def analyze_regime_profitability(self, df_combined: pd.DataFrame, symbol: str, interval: str, output_dir: Path, horizons: List[int]):
        """
        Analyzes future NET percentage returns over specified horizons, grouped by
        volatility regime and label.

        Assumes df_combined contains 'close', 'label', and 'volatility_regime' columns
        and is indexed by time.
        """
        self.logger.info(f"Starting Volatility Regime Net Profitability Analysis for {symbol.upper()} {interval} over horizons {horizons} bars...")

        required_cols = ['close', 'label', 'volatility_regime']
        if not all(col in df_combined.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df_combined.columns]
            self.logger.error(f"Missing required columns for Volatility Regime Profitability analysis: {missing}. Skipping.")
            return

        df_combined = df_combined.copy() # Work on a copy
        # Ensure close, label, and volatility_regime columns are numeric, coercing errors to NaN
        for col in required_cols:
            if col in df_combined.columns:
                df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

        # Drop rows where critical columns became NaN
        df_combined.dropna(subset=required_cols, inplace=True)

        if df_combined.empty:
            self.logger.warning("DataFrame is empty after dropping NaNs in critical columns. Skipping Volatility Regime Profitability analysis.")
            return

        # Ensure label and volatility_regime are integer types after dropping NaNs
        df_combined['label'] = df_combined['label'].astype(int)
        df_combined['volatility_regime'] = df_combined['volatility_regime'].astype(int)


        returns_data = []

        # The logic for Future Returns analysis (fixed horizon) remains the same.
        # It's about the potential return if you entered at a signal and held for exactly 'horizon' bars.
        # The '0' as hold logic within the f_window is not directly relevant here,
        # as this analysis is about the *potential* return at a fixed future point,
        # not the actual trade management.
        for horizon in horizons:
            future_close_series = df_combined['close'].shift(-horizon)

            for i in range(len(df_combined)):
                current_label = df_combined['label'].iloc[i]
                current_close = df_combined['close'].iloc[i]
                future_close = future_close_series.iloc[i]
                volatility_regime = df_combined['volatility_regime'].iloc[i]


                # Only consider points where a non-neutral signal was given as an "entry" for this analysis
                if current_label != 0 and pd.notna(current_close) and pd.notna(future_close) and abs(current_close) > FLOAT_EPSILON:
                    net_return = self._calculate_net_return_scalar(current_close, future_close, current_label) # Use 'current_label' as trade_type
                    if pd.notna(net_return):
                        returns_data.append({
                            'label': current_label,
                            'volatility_regime': volatility_regime,
                            'horizon': horizon,
                            'return_pct': net_return
                        })


        if not returns_data:
            self.logger.info("No non-zero labels with valid future net returns and volatility regimes found for analysis.")
            return

        df_regime_returns = pd.DataFrame(returns_data)

        self.logger.info("Calculating Volatility Regime Net Profitability summary statistics...")
        if len(df_regime_returns) > 0:
            agg_funcs = ['count', 'mean', 'median', 'std', 'min', 'max']
            agg_funcs_with_quantiles = agg_funcs + [lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]

            regime_returns_summary = df_regime_returns.groupby(['volatility_regime', 'label', 'horizon'])['return_pct'].agg(agg_funcs_with_quantiles).reset_index()
            regime_returns_summary.columns = ['Volatility Regime', 'Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile']

        else:
            self.logger.warning("No volatility regime returns data to calculate summary statistics.")
            regime_returns_summary = pd.DataFrame(columns=['Volatility Regime', 'Label', 'Horizon', 'Count', 'Mean Return', 'Median Return', 'Std Dev Return', 'Min Return', 'Max Return', '25th Percentile', '75th Percentile'])


        self.logger.info("\nVolatility Regime Net Profitability Summary (%) by Regime, Label, and Horizon:")
        self.logger.info(regime_returns_summary.to_string())

        analysis_type_suffix = "regime_net_profitability_summary"
        table_path = output_dir / self.table_pattern.format(
            symbol=symbol.upper(), interval=interval, analysis_type=analysis_type_suffix
        ).replace(':', '_')
        try:
            regime_returns_summary.to_csv(table_path, index=False)
            self.logger.info(f"Saved volatility regime net profitability summary to {table_path}")
        except Exception as e:
            self.logger.error(f"Failed to save volatility regime net profitability summary table: {e}")

        self.logger.info("Generating Volatility Regime Net Profitability plots...")
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

                # Manually create legend handles and labels to ensure correctness
                handles, labels = ax.get_legend_handles_labels()
                legend_labels_map = {'-1': 'Short (-1)', '1': 'Long (1)'}

                # Filter and map handles and labels
                final_handles = []
                final_labels = []
                # Iterate through expected labels in a desired order
                for label_val in [-1, 1]:
                    label_str = str(label_val)
                    if label_str in labels: # Check if this label is actually present in the plot
                        idx = labels.index(label_str)
                        final_handles.append(handles[idx])
                        final_labels.append(legend_labels_map.get(label_str, label_str)) # Use map or fallback to original

                # Place the legend only if there are items to show
                if final_handles:
                    plt.legend(final_handles, final_labels, title='Label') # Corrected order for legend display

                plt.tight_layout()

                plot_type_suffix = f"regime_{regime}_net_profitability_plot"
                plot_path = output_dir / self.plot_pattern.format(
                    symbol=symbol.upper(), interval=interval, analysis_type=plot_type_suffix
                ).replace(':', '_')
                try:
                    plt.savefig(plot_path, dpi=150)
                    self.logger.info(f"Saved volatility regime profitability plot (Regime {regime}) to {plot_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save volatility regime profitability plot (Regime {regime}): {e}")


        except ImportError:
            self.logger.warning("Matplotlib or Seaborn not installed. Skipping plot generation.")
        except Exception as e:
            self.logger.error(f"Error plotting volatility regime profitability: {e}", exc_info=True)
        finally:
            if 'plt' in locals() and plt.get_fignums():
                 plt.close('all')


        self.logger.info("Volatility Regime Net Profitability Analysis complete.")


    def perform_all_analyses(self, df_combined: pd.DataFrame, symbol: str, interval: str, label_strategy: str, future_horizons: List[int]):
        """
        Orchestrates all label analyses and saves results to a strategy-specific folder.

        Args:
            df_combined (pd.DataFrame): Combined DataFrame with OHLCV, features, and 'label' column.
            symbol (str): Trading pair symbol.
            interval (str): Time interval.
            label_strategy (str): The name of the labeling strategy being analyzed.
            future_horizons (List[int]): List of future horizons for return analysis.
        """
        self.logger.info(f"Performing all analyses for {symbol} {interval} with strategy '{label_strategy}'...")

        # Dynamically create the strategy-specific analysis directory
        analysis_output_dir = Path(self.strategy_dir_pattern_str.format(label_strategy=label_strategy))
        try:
            analysis_output_dir.mkdir(exist_ok=True, parents=True)
            self.logger.info(f"Ensured analysis results directory exists: {analysis_output_dir}")
        except OSError as e:
            self.logger.error(f"Error creating analysis results directory {analysis_output_dir}: {e}", exc_info=True)
            raise # Re-raise to stop pipeline if directory cannot be created

        # 0. Overall Label Distribution Analysis (NEW)
        self.analyze_label_distribution(df_combined.copy(), symbol, interval, analysis_output_dir)

        # 1. Label Streak Analysis
        self.analyze_label_streaks(df_combined.copy(), symbol, interval, analysis_output_dir)

        # 2. Max Favorable Return / Max Adverse Loss Analysis (formerly MFE/MAE)
        # This function now returns the DataFrame containing the individual MFR/MAL values
        # which is needed for the ratio analysis.
        df_max_return_loss_for_ratio = self.analyze_mfe_mae(df_combined.copy(), symbol, interval, analysis_output_dir)

        # 2.1. Max Return/Loss Ratio Analysis (NEW) - REMOVED AS REQUESTED
        # if df_max_return_loss_for_ratio is not None and not df_max_return_loss_for_ratio.empty:
        #     self.analyze_max_return_loss_ratio(df_max_return_loss_for_ratio, symbol, interval, analysis_output_dir)
        # else:
        #     self.logger.warning("Skipping Max Return/Loss Ratio Analysis as no valid MFR/MAL data was generated.")


        # 3. Future Returns Analysis (General)
        self.analyze_future_returns(df_combined.copy(), symbol, interval, analysis_output_dir, future_horizons)

        # 4. Volatility Regime Profitability Analysis
        self.analyze_regime_profitability(df_combined.copy(), symbol, interval, analysis_output_dir, future_horizons)

        self.logger.info(f"All analyses completed for {symbol} {interval} with strategy '{label_strategy}'.")

