# utils/analysis/metrics_calculator.py

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from config.params import AppConfig, FLOAT_EPSILON

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """
    Calculates a comprehensive set of performance metrics for trading strategies.
    Designed to be used by both PerformanceAnalyzer and MonteCarloAnalyzer.
    """
    def __init__(self, app_config: AppConfig, initial_capital: float):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.app_config = app_config
        self.initial_capital = initial_capital

        self.bars_per_year: Optional[int] = None
        # This will be set by the calling analyzer, or defaulted
        # It's not determined in init because interval is not always known here

    def calculate_all_metrics(self, trade_history_df: pd.DataFrame, equity_df: pd.DataFrame, interval: str) -> Dict[str, Any]:
        """
        Calculates a comprehensive set of performance metrics.
        """
        self.logger.info(f"\n{'-'*30}\n📏 Calculating Summary Metrics\n{'-'*30}")
        metrics: Dict[str, Any] = {}

        # Set bars_per_year based on the interval passed
        # CORRECTED: Access bars_per_year from app_config.trading
        self.bars_per_year = self.app_config.trading.bars_per_year
        if self.bars_per_year is None: # Fallback if for some reason it's still None
            self.logger.warning(f"bars_per_year not found in app_config.trading. Falling back to 24*365.")
            self.bars_per_year = 24 * 365 # Default fallback to hourly for a year


        # Handle empty equity curve scenarios
        if equity_df.empty or 'equity' not in equity_df.columns or equity_df['equity'].empty:
            self.logger.warning("⚠️ Equity curve is empty or invalid. Cannot calculate most performance metrics.")
            metrics["Error"] = "No equity curve to calculate metrics."
            metrics.update({
                'Total Return (%)': np.nan, 'CAGR (%)': np.nan, 'Sharpe Ratio': np.nan,
                'Sortino Ratio': np.nan, 'Max Drawdown (%)': np.nan, 'Final Equity': np.nan,
                'Peak Equity': np.nan, 'Equity Change': np.nan
            })
            metrics.update(self._get_trade_metrics(pd.DataFrame(), interval))
            metrics["PnL Consistency Check"] = "Unavailable"
            return metrics

        # Ensure equity curve is sorted by index (time)
        equity_df = equity_df.sort_index()

        # Basic Equity Metrics
        final_equity = equity_df['equity'].iloc[-1]
        metrics['Final Equity'] = f"{final_equity:.2f}"
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100 if self.initial_capital != 0 else 0
        metrics['Total Return (%)'] = total_return
        
        # Returns for CAGR, Sharpe, Sortino, etc.
        if len(equity_df) < 2:
            self.logger.warning("Equity curve too short for return calculations. Skipping annualized metrics.")
            metrics.update({
                'CAGR (%)': np.nan, 'Sharpe Ratio': np.nan, 'Sortino Ratio': np.nan
            })
            returns = pd.Series(dtype=float)
        else:
            returns = equity_df['equity'].pct_change().dropna()
            
            if self.bars_per_year is None or self.bars_per_year <= 0:
                self.logger.warning(f"Annualization factor ({self.bars_per_year}) is invalid. Cannot calculate annualized metrics.")
                metrics.update({
                    'CAGR (%)': np.nan, 'Sharpe Ratio': np.nan, 'Sortino Ratio': np.nan
                })
            elif returns.empty or returns.std() < FLOAT_EPSILON:
                self.logger.warning("Returns series is empty or has zero standard deviation. Cannot calculate Sharpe/Sortino.")
                metrics.update({
                    'CAGR (%)': np.nan, 'Sharpe Ratio': np.nan, 'Sortino Ratio': np.nan
                })
            else:
                # CAGR (Compound Annual Growth Rate)
                total_duration = (equity_df.index[-1] - equity_df.index[0]).total_seconds()
                total_duration_years = total_duration / (365.25 * 24 * 3600)
                
                if total_duration_years > FLOAT_EPSILON and self.initial_capital > FLOAT_EPSILON:
                    cagr = ((final_equity / self.initial_capital) ** (1 / total_duration_years) - 1) * 100
                    metrics['CAGR (%)'] = cagr
                else:
                    metrics['CAGR (%)'] = np.nan if self.initial_capital <= FLOAT_EPSILON else 0.0

                # Sharpe Ratio
                risk_free_rate = 0.0 # Assume risk-free rate is 0
                excess_returns = returns - risk_free_rate
                sharpe_ratio = (excess_returns.mean() / (returns.std() + FLOAT_EPSILON)) * np.sqrt(self.bars_per_year)
                metrics['Sharpe Ratio'] = sharpe_ratio

                # Sortino Ratio
                downside_returns = returns[returns < 0]
                if not downside_returns.empty and downside_returns.std() > FLOAT_EPSILON:
                    sortino_ratio = (excess_returns.mean() / (downside_returns.std() + FLOAT_EPSILON)) * np.sqrt(self.bars_per_year)
                    metrics['Sortino Ratio'] = sortino_ratio
                else:
                    metrics['Sortino Ratio'] = np.nan # No downside returns or zero std

        # Drawdown Metrics
        rolling_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - rolling_max) / (rolling_max + FLOAT_EPSILON) * 100
        max_drawdown = drawdown.min()
        metrics['Max Drawdown (%)'] = max_drawdown
        metrics['Peak Equity'] = f"{equity_df['equity'].max():.2f}"
        metrics['Equity Change'] = f"{final_equity - self.initial_capital:.2f}"


        # Trade-specific Metrics
        metrics.update(self._get_trade_metrics(trade_history_df, interval))

        # PnL Consistency Check
        metrics.update({"PnL Consistency Check": self._check_pnl_consistency(metrics)})

        self.logger.info("✅ Summary metrics calculated successfully.")
        return metrics

    def _get_trade_metrics(self, trades_df: pd.DataFrame, interval: str) -> Dict[str, Any]:
        """
        Helper to calculate trade-specific metrics from a DataFrame of trades.
        """
        metrics = {}
        num_trades = len(trades_df)
        metrics["Number of Trades"] = num_trades

        # Initialize holding duration variables to np.nan
        avg_holding_duration_minutes = np.nan
        max_holding_duration_minutes = np.nan

        if num_trades > 0 and 'net_pnl' in trades_df.columns:
            metrics["Total Net PnL (Sum Trades)"] = trades_df['net_pnl'].sum()
            metrics["Total Gross PnL (Sum Trades)"] = trades_df['gross_pnl'].sum() if 'gross_pnl' in trades_df.columns else np.nan
            metrics["Total Fees Paid"] = trades_df['total_fees'].sum() if 'total_fees' in trades_df.columns else np.nan

            metrics["Number of Wins"] = (trades_df['net_pnl'] > FLOAT_EPSILON).sum()
            metrics["Number of Losses"] = (trades_df['net_pnl'] < -FLOAT_EPSILON).sum()
            metrics["Number of Breakeven"] = num_trades - metrics["Number of Wins"] - metrics["Number of Losses"]

            metrics["Win Rate (%)"] = (metrics["Number of Wins"] / num_trades) * 100 if num_trades > 0 else 0.0
            metrics["Loss Rate (%)"] = (metrics["Number of Losses"] / num_trades) * 100 if num_trades > 0 else 0.0
            metrics["Breakeven Rate (%)"] = (metrics["Number of Breakeven"] / num_trades) * 100 if num_trades > 0 else 0.0
            
            avg_pnl_per_trade = metrics["Total Net PnL (Sum Trades)"] / num_trades if num_trades > 0 else np.nan
            metrics["Avg PnL per Trade"] = avg_pnl_per_trade

            wins_pnl = trades_df.loc[trades_df['net_pnl'] > FLOAT_EPSILON, 'net_pnl']
            losses_pnl = trades_df.loc[trades_df['net_pnl'] < -FLOAT_EPSILON, 'net_pnl']

            metrics["Average Win"] = wins_pnl.mean() if not wins_pnl.empty else 0.0
            metrics["Average Loss"] = abs(losses_pnl.mean()) if not losses_pnl.empty else 0.0 # Store as positive magnitude

            # Profit Factor: (Sum of Winning PnL) / (Absolute Sum of Losing PnL)
            total_winning_pnl = wins_pnl.sum()
            total_losing_pnl = abs(losses_pnl.sum())
            metrics["Profit Factor"] = total_winning_pnl / (total_losing_pnl + FLOAT_EPSILON) if total_losing_pnl > FLOAT_EPSILON else (np.inf if total_winning_pnl > FLOAT_EPSILON else 0)

            # Edge (Expected Value) per trade
            win_rate_fraction = metrics["Win Rate (%)"] / 100.0
            avg_win = metrics["Average Win"]
            avg_loss = metrics["Average Loss"] # This is already positive magnitude
            metrics["Edge (Expected Value)"] = (win_rate_fraction * avg_win) - ((1 - win_rate_fraction) * avg_loss)


            # Duration Metrics
            minutes_per_bar = np.nan
            if interval.endswith('m'):
                minutes_per_bar = float(interval.replace('m', ''))
            elif interval.endswith('h'):
                minutes_per_bar = float(interval.replace('h', '')) * 60
            elif interval.endswith('d'):
                minutes_per_bar = float(interval.replace('d', '')) * 24 * 60
            elif interval.endswith('w'):
                minutes_per_bar = float(interval.replace('w', '')) * 7 * 24 * 60
            elif interval.endswith('M'): # Monthly approx.
                minutes_per_bar = float(interval.replace('M', '')) * 30 * 24 * 60 # Approx 30 days
            
            if 'holding_duration_seconds' in trades_df.columns and not trades_df['holding_duration_seconds'].empty:
                avg_holding_duration_minutes = trades_df['holding_duration_seconds'].mean() / 60.0
                max_holding_duration_minutes = trades_df['holding_duration_seconds'].max() / 60.0
            elif 'holding_period_bars' in trades_df.columns and not trades_df['holding_period_bars'].empty and pd.notna(minutes_per_bar):
                avg_holding_duration_minutes = trades_df['holding_period_bars'].mean() * minutes_per_bar
                max_holding_duration_minutes = trades_df['holding_period_bars'].max() * minutes_per_bar
            else:
                self.logger.warning("Cannot calculate average/max holding duration: relevant columns or interval parsing invalid.")

            metrics["Avg Holding Duration (minutes)"] = avg_holding_duration_minutes if pd.notna(avg_holding_duration_minutes) else np.nan
            metrics["Max Holding Duration (minutes)"] = max_holding_duration_minutes if pd.notna(max_holding_duration_minutes) else np.nan


        else:
            # Default values for empty trades_df
            metrics.update({
                "Total Net PnL (Sum Trades)": 0.0, "Total Gross PnL (Sum Trades)": 0.0, "Total Fees Paid": 0.0,
                "Number of Wins": 0, "Number of Losses": 0, "Number of Breakeven": 0,
                "Win Rate (%)": 0.0, "Loss Rate (%)": 0.0, "Breakeven Rate (%)": 0.0,
                "Avg PnL per Trade": np.nan, "Average Win": 0.0, "Average Loss": 0.0,
                "Edge (Expected Value)": 0.0, "Profit Factor": np.nan,
                "Avg Holding Duration (minutes)": np.nan, "Max Holding Duration (minutes)": np.nan
            })
        return metrics

    def _check_pnl_consistency(self, metrics: Dict[str, Any]) -> str:
        """
        Checks if the final equity change matches the sum of net PnLs from trades.
        """
        try:
            final_equity = float(metrics.get('Final Equity', self.initial_capital))
        except (ValueError, TypeError):
            final_equity = self.initial_capital

        equity_change = final_equity - self.initial_capital
        
        try:
            total_net_pnl_from_trades = float(metrics.get('Total Net PnL (Sum Trades)', 0.0))
        except (ValueError, TypeError):
            total_net_pnl_from_trades = 0.0

        if pd.notna(equity_change) and pd.notna(total_net_pnl_from_trades):
            discrepancy = abs(equity_change - total_net_pnl_from_trades)
            tolerance = max(abs(self.initial_capital * 1e-6), 1e-4)

            if discrepancy > tolerance:
                self.logger.critical(f"\n{'!'*30}\n❌ CRITICAL PNL DISCREPANCY DETECTED!\nEquity Change: {equity_change:.6f}\nSum of Trade Net PnLs: {total_net_pnl_from_trades:.6f}\nDiscrepancy: {discrepancy:.6f}\n{'!'*30}")
                return f"FAIL (Discrepancy: {discrepancy:.6f})"
            else:
                self.logger.info(f"✅ PnL Consistency Check Passed (Discrepancy: {discrepancy:.6f})")
                return "PASS"
        else:
            self.logger.warning("⚠️ Could not perform PnL consistency check due to missing/invalid metrics.")
            return "Unavailable"

