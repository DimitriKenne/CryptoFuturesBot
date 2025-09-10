import pandas as pd
import numpy as np
from .base_strategy import BaseLabelingStrategy

def attach_htf_context(df: pd.DataFrame, htf_interval: str = "1h") -> pd.DataFrame:
    """
    Attach HTF open, close, and HTF bar index for each LTF bar.
    Assumes df has a DatetimeIndex named 'timestamp' and LTF OHLC columns.
    Returns df with added columns: 'htf_open', 'htf_close', 'htf_bar_index'
    """
    # Resample to HTF
    htf_ohlc = df.resample(htf_interval).agg(
        htf_open=('open', 'first'),
        htf_close=('close', 'last')
    )
    # Assign an integer HTF bar index
    htf_ohlc['htf_bar_index'] = np.arange(len(htf_ohlc))

    # Map each LTF bar to its HTF context
    df = df.copy()
    # Find for each LTF bar the latest HTF timestamp <= its own timestamp
    htf_map = pd.Series(htf_ohlc.index, index=htf_ohlc.index)
    ltf_to_htf = df.index.map(lambda ts: htf_map[htf_map <= ts].max())
    df['htf_open'] = htf_ohlc.loc[ltf_to_htf, 'htf_open'].values
    df['htf_close'] = htf_ohlc.loc[ltf_to_htf, 'htf_close'].values
    df['htf_bar_index'] = htf_ohlc.loc[ltf_to_htf, 'htf_bar_index'].values
    return df

class Strategy5(BaseLabelingStrategy):
    """
    HTF Context Return labeling.
    Label LTF bar as 1/-1 if, in the HTF, no opposite bar appears before % net return is attained,
    using entry and exit prices adjusted for trading fee and slippage.
    """

    def calculate_raw_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        # Attach HTF context columns, drop them at the end
        htf_interval = getattr(self.config, "htf_interval", "1h")
        df = attach_htf_context(df, htf_interval=htf_interval)
        out = df.copy()
        out['label'] = 0

        return_thresh = self.config.return_threshold_pct / 100.0
        trading_fee = self.trading_fee_rate
        slippage = self.slippage_tolerance_rate

        n = len(df)
        for i in range(n):
            htf_idx = int(df['htf_bar_index'].iloc[i])
            # Get HTF context for current LTF bar
            htf_open = df['htf_open'].iloc[i]
            htf_close = df['htf_close'].iloc[i]
            htf_bull = htf_close > htf_open
            htf_bear = htf_close < htf_open

            if htf_bull:
                entry_price = df['close'].iloc[i] * (1 + trading_fee + slippage)
                for j in range(i + 1, n):
                    # If HTF context changes, abort
                    if int(df['htf_bar_index'].iloc[j]) != htf_idx or df['htf_close'].iloc[j] < df['htf_open'].iloc[j]:
                        break
                    exit_price = df['high'].iloc[j] * (1 - trading_fee - slippage)
                    realized_return = (exit_price - entry_price) / entry_price
                    if realized_return >= return_thresh:
                        out.iloc[i, out.columns.get_loc('label')] = 1
                        break
            elif htf_bear:
                entry_price = df['close'].iloc[i] * (1 - trading_fee - slippage)
                for j in range(i + 1, n):
                    # If HTF context changes, abort
                    if int(df['htf_bar_index'].iloc[j]) != htf_idx or df['htf_close'].iloc[j] > df['htf_open'].iloc[j]:
                        break
                    exit_price = df['low'].iloc[j] * (1 + trading_fee + slippage)
                    realized_return = (entry_price - exit_price) / entry_price
                    if realized_return >= return_thresh:
                        out.iloc[i, out.columns.get_loc('label')] = -1
                        break
        # Drop HTF columns for cleanliness
        return out[['label']]

    def perform_strategy_specific_analysis(self, df_original_input, plotter, calculator):
        return []