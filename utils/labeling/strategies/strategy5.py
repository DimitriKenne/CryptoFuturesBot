import pandas as pd
import numpy as np
from .base_strategy import BaseLabelingStrategy

def attach_htf_context(df: pd.DataFrame, htf_interval: str = "1d") -> pd.DataFrame:
    """
    Attach HTF open, close, high, low, and HTF bar index for each LTF bar using merge_asof.
    Assumes df has a DatetimeIndex named 'timestamp' and LTF OHLC columns.
    """
    htf_ohlc = df.resample(htf_interval).agg(
        htf_open=('open', 'first'),
        htf_close=('close', 'last'),
        htf_high=('high', 'max'),
        htf_low=('low', 'min'),
    ).reset_index()
    htf_ohlc['htf_bar_index'] = np.arange(len(htf_ohlc))
    # Ensure both indexes are sorted
    df_sorted = df.sort_index().reset_index()
    result = pd.merge_asof(
        df_sorted,
        htf_ohlc,
        left_on='timestamp',
        right_on='timestamp',
        direction='backward'
    )
    result.set_index('timestamp', inplace=True)
    return result

class Strategy5(BaseLabelingStrategy):
    """
    HTF-LTF Context Quantile labeling using correct formulas:
    - Long trades use HTF/LTF open to HTF high.
    - Short trades use HTF/LTF open to HTF low.
    - Quantiles use all returns.
    """

    def __init__(self, config, logger, trading_fee_rate, slippage_tolerance_rate):
        super().__init__(config, logger, trading_fee_rate, slippage_tolerance_rate)
        self._intermediate_htf_returns_df = pd.DataFrame()

    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        htf_interval = getattr(self.config, "htf_timeframe", "1d")
        df = attach_htf_context(df, htf_interval=htf_interval)
        out = df.copy()
        out['label'] = 0

        trading_fee = self.trading_fee_rate
        slippage = self.slippage_tolerance_rate

        # --- Compute HTF (daily) returns (open to high/low, with fee/slippage) ---
        htf_df = (
            df[['htf_bar_index', 'htf_open', 'htf_high', 'htf_low']]
            .drop_duplicates('htf_bar_index')
            .set_index('htf_bar_index')
        )
        htf_entry_long = htf_df['htf_open'] * (1 + trading_fee + slippage)
        htf_exit_long = htf_df['htf_high'] * (1 - trading_fee - slippage)
        htf_net_return_long = (htf_exit_long - htf_entry_long) / htf_entry_long

        htf_entry_short = htf_df['htf_open'] * (1 - trading_fee - slippage)
        htf_exit_short = htf_df['htf_low'] * (1 + trading_fee + slippage)
        htf_net_return_short = (htf_entry_short - htf_exit_short) / htf_entry_short

        # Store for analysis
        self._intermediate_htf_returns_df = pd.DataFrame({
            "Net_Return_Long": htf_net_return_long,
            "Net_Return_Short": htf_net_return_short
        })

        # --- Quantiles: Use all returns ---
        bullish_pct = getattr(self.config, "bullish_quantile_pct", 75.0)
        bearish_pct = getattr(self.config, "bearish_quantile_pct", 25.0)
        bullish_threshold = np.percentile(htf_net_return_long, bullish_pct) if len(htf_net_return_long) > 0 else 0.0
        bearish_threshold = np.percentile(htf_net_return_short, bearish_pct) if len(htf_net_return_short) > 0 else 0.0

        # --- HTF Regime Labels ---
        htf_labels = pd.Series(0, index=htf_df.index)
        htf_labels[htf_net_return_long >= bullish_threshold] = 1
        htf_labels[htf_net_return_short <= bearish_threshold] = -1

        # --- LTF Returns: Compute returns from LTF open to HTF high/low ---
        ltf_entry_long = df['open'] * (1 + trading_fee + slippage)
        ltf_exit_long = df['htf_high'] * (1 - trading_fee - slippage)
        ltf_net_return_long = (ltf_exit_long - ltf_entry_long) / ltf_entry_long

        ltf_entry_short = df['open'] * (1 - trading_fee - slippage)
        ltf_exit_short = df['htf_low'] * (1 + trading_fee + slippage)
        ltf_net_return_short = (ltf_entry_short - ltf_exit_short) / ltf_entry_short

        # Propagate HTF label to LTF bars
        ltf_htf_label = df['htf_bar_index'].map(htf_labels).values

        # --- LTF Labeling ---
        label_arr = np.zeros(len(df), dtype=int)
        bullish_mask = ltf_htf_label == 1
        label_arr[bullish_mask & (ltf_net_return_long >= bullish_threshold)] = 1

        bearish_mask = ltf_htf_label == -1
        label_arr[bearish_mask & (ltf_net_return_short <= bearish_threshold)] = -1

        out['label'] = label_arr

        # Drop HTF columns for cleanliness
        return out[['label']]

    def perform_strategy_specific_analysis(self, df_original_input, plotter, calculator):
        # Return net return distributions for analysis, similar to strategy2
        df_net_returns = self._intermediate_htf_returns_df.copy()
        df_net_returns['Net_Return_Long'] *= 100.0
        df_net_returns['Net_Return_Short'] *= 100.0

        if df_net_returns.empty:
            return []

        fig = plotter.plot_net_return_distributions(df_net_returns)
        return [("net_return_distributions", fig)]