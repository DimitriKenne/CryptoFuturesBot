# utils/strategy_execution/backtester.py

import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Literal

# Add project root to Python path for imports if needed in a standalone run
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent # Adjust based on actual depth
sys.path.append(str(PROJECT_ROOT))


# Import all modular components
try:
    from config.params import AppConfig # Ensure AppConfig is imported
    from config.paths import PATHS # For general path lookups
    from utils.data_management.market_data_handler import MarketDataHandler
    from utils.strategy_execution.trade_execution_engine import TradeExecutionEngine
    from utils.strategy_execution.trading_session_manager import TradingSessionManager
    from utils.analysis.performance_analyzer import PerformanceAnalyzer # Renamed from ResultsAnalyser
except ImportError as e:
    logging.critical(f"Failed to import necessary modules for Backtester: {e}. Ensure all config and utils files are correctly placed.", exc_info=True)
    sys.exit(1)

logger = logging.getLogger(__name__)

class Backtester:
    """
    Orchestrates the backtesting process by simulating trade execution over historical data.
    It integrates MarketDataHandler, TradeExecutionEngine, TradingSessionManager,
    and PerformanceAnalyzer to provide a comprehensive backtesting solution.
    """

    def __init__(self,
                 app_config: AppConfig,
                 symbol: str,
                 interval: str,
                 model_type: str,
                 backtest_mode: Literal['full', 'train', 'test'],
                 train_ratio: float = 0.7,
                 initial_ohlcv_data: Optional[pd.DataFrame] = None # For Monte Carlo or specific test sets
                 ):
        """
        Initializes the Backtester.

        Args:
            app_config (AppConfig): The global application configuration object.
            symbol (str): The trading pair symbol (e.g., 'BTCUSDT').
            interval (str): The data interval (e.g., '1h').
            model_type (str): The type of ML model used (e.g., 'xgboost', 'lstm').
            backtest_mode (Literal['full', 'train', 'test']): The mode for data splitting.
            train_ratio (float): The ratio of data to use for training (0.0 to 1.0).
                                 Only applicable in "train" or "test" backtest_mode.
            initial_ohlcv_data (Optional[pd.DataFrame]): Optional, pre-loaded OHLCV data.
                                                        If provided, data loading from files is skipped.
                                                        Used for Monte Carlo simulations.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing Backtester for {symbol}-{interval} with model {model_type} in {backtest_mode} mode.")

        self.app_config = app_config
        self.symbol = symbol.upper()
        self.interval = interval
        self.model_type = model_type
        self.backtest_mode = backtest_mode
        self.train_ratio = train_ratio
        self.initial_ohlcv_data = initial_ohlcv_data


        # Initialize core components using the provided app_config
        # MarketDataHandler handles data loading, feature engineering, and signal generation
        self.market_data_handler = MarketDataHandler(
            mode='backtest', # This mode is for overall behavior, not splitting
            symbol=self.symbol,
            interval=self.interval,
            model_type=self.model_type,
            app_config=self.app_config, # Pass app_config to MarketDataHandler
            initial_ohlcv_data=self.initial_ohlcv_data, # Pass initial data if provided
            train_ratio=self.train_ratio,
            backtest_mode=self.backtest_mode # PASSED HERE: for data splitting when loading from file
        )

        # TradeExecutionEngine handles position sizing, SL/TP, filters, PnL
        self.trade_execution_engine = TradeExecutionEngine(app_config=self.app_config)

        # TradingSessionManager tracks capital, open positions, and trade history
        self.trading_session_manager = TradingSessionManager(
            # Removed initial_capital=self.app_config.trading.risk.initial_capital
            app_config=self.app_config # Pass app_config to TradingSessionManager
        )

        self.logger.info("Backtester initialized. Ready to run simulation.")


    def run_backtest(self) -> Tuple[pd.DataFrame, pd.Series, 'PerformanceAnalyzer']:
        """
        Executes the backtest simulation over the historical data.

        Returns:
            Tuple[pd.DataFrame, pd.Series, PerformanceAnalyzer]:
                - final_trade_history (pd.DataFrame): DataFrame of all completed trades.
                - final_equity_curve (pd.Series): Series representing the account equity over time.
                - performance_analyzer (PerformanceAnalyzer): Initialized analyzer for further metrics/plots.
        """
        self.logger.info(f"\n{'='*40}\n🚀 Starting Backtest: {self.symbol} {self.interval} ({self.model_type})\n{'='*40}")

        processed_bar_data = None

        try:
            data_stream_generator = self.market_data_handler.get_processed_data_stream()
            
            for i, processed_bar_data in enumerate(data_stream_generator):
                if processed_bar_data is None:
                    self.logger.warning(f"[BAR {i}] No data, skipping.")
                    continue

                current_timestamp = processed_bar_data.name
                current_price = processed_bar_data['close']
                signal = processed_bar_data.get('signal', 0)
                model_probabilities = processed_bar_data.get('probabilities', {})

                if i == 0:
                    self.trading_session_manager.initialize_equity_curve(current_timestamp, self.app_config.trading.risk.initial_capital)

                # Ensure model_probabilities is a pd.Series
                if isinstance(model_probabilities, dict):
                    model_probabilities = pd.Series(model_probabilities)
                elif isinstance(model_probabilities, list) and len(model_probabilities) > 0 and isinstance(model_probabilities[0], dict):
                    model_probabilities = pd.Series(model_probabilities[0])
                elif not isinstance(model_probabilities, pd.Series):
                    model_probabilities = pd.Series(dtype=float)

                open_trade = self.trading_session_manager.get_open_position()
                exit_triggered = False

                # --- 1. Manage Open Positions (Check for Exits) ---
                if open_trade:
                    self.logger.info(
                        f"[BAR {i}] 🟡 Open {open_trade['direction_str'].upper()} | Entry: {open_trade['entry_price']:.4f} | Qty: {open_trade['quantity']:.2f} | SL: {open_trade['stop_loss_price']:.4f} | TP: {open_trade['take_profit_price']:.4f}"
                    )
                    exit_triggered, exit_reason, exit_price = self.trade_execution_engine.check_exit_conditions(
                        open_trade=open_trade,
                        current_bar_data=processed_bar_data,
                        current_bar_index=i
                    )
                    if exit_triggered:
                        self.logger.info(
                            f"[BAR {i}] 🔴 EXIT: {exit_reason.upper()} @ {exit_price:.4f} (Entry: {open_trade['entry_price']:.4f}, Qty: {open_trade['quantity']:.2f})"
                        )
                        completed_trade_details = self.trade_execution_engine.calculate_exit_details(
                            open_trade=open_trade,
                            exit_price=exit_price,
                            exit_time=current_timestamp,
                            exit_reason=exit_reason
                        )
                        self.trading_session_manager.close_position(completed_trade_details)
                        self.logger.info(
                            f"[BAR {i}] ✅ Trade closed. NetPnL: {completed_trade_details.get('net_pnl', 0.0):.2f}, Fees: {completed_trade_details.get('entry_fee', 0.0) + completed_trade_details.get('exit_fee', 0.0):.2f}, Capital: {self.trading_session_manager.get_current_capital():.2f}"
                        )

                # --- 2. Soft Exit on Neutral Signal ---
                if open_trade and not exit_triggered and self.app_config.trading.trade_execution.exit_on_neutral_signal and signal == 0:
                    self.logger.info(
                        f"[BAR {i}] 🔵 EXIT: NEUTRAL_SIGNAL @ {current_price:.4f} (Entry: {open_trade['entry_price']:.4f}, Qty: {open_trade['quantity']:.2f})"
                    )
                    completed_trade_details = self.trade_execution_engine.calculate_exit_details(
                        open_trade=open_trade,
                        exit_price=current_price,
                        exit_time=current_timestamp,
                        exit_reason='neutral_signal'
                    )
                    self.trading_session_manager.close_position(completed_trade_details)
                    self.logger.info(
                        f"[BAR {i}] ✅ Trade closed. NetPnL: {completed_trade_details.get('net_pnl', 0.0):.2f}, Fees: {completed_trade_details.get('entry_fee', 0.0) + completed_trade_details.get('exit_fee', 0.0):.2f}, Capital: {self.trading_session_manager.get_current_capital():.2f}"
                    )

                # --- 3. Entry Logic ---
                if not self.trading_session_manager.get_open_position() and signal != 0:
                    entry_details = self.trade_execution_engine.calculate_entry_details(
                        signal=signal,
                        current_capital=self.trading_session_manager.get_current_equity(),
                        current_price=current_price,
                        current_bar_features=processed_bar_data,
                        model_probabilities=model_probabilities,
                        current_bar_index=i
                    )
                    if entry_details:
                        self.logger.info(
                            f"[BAR {i}] 🟢 ENTRY: {entry_details['direction_str'].upper()} @ {entry_details['entry_price']:.4f} | Qty: {entry_details['quantity']:.2f} | SL: {entry_details['stop_loss_price']:.4f} | TP: {entry_details['take_profit_price']:.4f}"
                        )
                        self.trading_session_manager.set_open_position(entry_details)

                # --- 4. Update Equity Curve (End of Bar) ---
                equity = self.trading_session_manager.get_current_equity(current_price=current_price)
                capital = self.trading_session_manager.get_current_capital()
                self.trading_session_manager.update_equity_curve(equity, current_timestamp)
                self.logger.info(f"[BAR {i}] 💰 Equity: {equity:.2f} | Capital: {capital:.2f}")

        except Exception as e:
            self.logger.critical(f"Backtest error: {e}", exc_info=True)
        finally:
            self.logger.info(f"\n{'='*40}\n🏁 Backtest Finished\n{'='*40}")
            final_trade_history = self.trading_session_manager.get_trade_history()
            final_equity_curve = self.trading_session_manager.get_equity_curve()

            # --- Close any open trade at the last bar price ---
            if self.trading_session_manager.get_open_position() and processed_bar_data is not None:
                open_trade = self.trading_session_manager.get_open_position()
                last_price_for_equity = processed_bar_data['close']
                last_timestamp = processed_bar_data.name
                self.logger.info(
                    f"Closing open trade at end of backtest: {open_trade['direction_str'].upper()} @ {last_price_for_equity:.4f} (forced exit)"
                )
                completed_trade_details = self.trade_execution_engine.calculate_exit_details(
                    open_trade=open_trade,
                    exit_price=last_price_for_equity,
                    exit_time=last_timestamp,
                    exit_reason='end_of_backtest'
                )
                self.trading_session_manager.close_position(completed_trade_details)
                final_trade_history = self.trading_session_manager.get_trade_history()
                final_equity_curve = self.trading_session_manager.get_equity_curve()

            self.logger.info(f"\n{'-'*30}\nFinal Equity: {final_equity_curve.iloc[-1]:.2f}\nTotal Trades: {len(final_trade_history)}\n{'-'*30}")
            self.logger.info(f"Equity Curve (last 10):\n{final_equity_curve.tail(10)}")


            # Instantiate PerformanceAnalyzer and return it
            performance_analyzer = PerformanceAnalyzer(
                symbol=self.symbol,
                interval=self.interval,
                model_type=self.model_type,
                initial_capital=self.app_config.trading.risk.initial_capital,
                trade_history_data=final_trade_history,
                equity_data=final_equity_curve,
                results_type='backtest',
                app_config=self.app_config
            )

            return final_trade_history, final_equity_curve, performance_analyzer

    def save_results(self, trades_df: Optional[pd.DataFrame] = None, equity_curve: Optional[pd.Series] = None):
        """
        Saves trades and equity curve to files based on PATHS config.
        Trades and equity curve are saved in results/backtesting/.
        If no arguments are provided, uses the current trade history and equity curve.
        """
        self.logger.info("Saving backtest results...")
        results_dir = PATHS.get("backtesting_results_dir")
        if not results_dir or not isinstance(results_dir, (str, Path)):
            self.logger.error("Cannot save results: 'backtesting_results_dir' invalid or missing in paths config.")
            return
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        trades_pattern = str(PATHS.get("backtesting_trades_pattern", "{symbol}_{interval}_{model_type}_trades.parquet"))
        equity_pattern = str(PATHS.get("backtesting_equity_pattern", "{symbol}_{interval}_{model_type}_equity.parquet"))

        file_params = {
            "symbol": self.symbol.replace('/', ''),
            "interval": self.interval.replace(':', '_'),
            "model_type": self.model_type
        }

        # Use current trade history and equity curve if not provided
        if trades_df is None:
            trades_df = self.trading_session_manager.get_trade_history()
        if isinstance(trades_df, list):
            trades_df = pd.DataFrame(trades_df)
        if equity_curve is None:
            equity_curve = self.trading_session_manager.get_equity_curve()

        try:
            # --- Save Trades ---
            if trades_df is not None and not trades_df.empty:
                trades_path = results_dir / trades_pattern.format(**file_params)
                # Ensure datetime columns are timezone-aware (UTC) before saving
                for col in ['entry_time', 'exit_time']:
                    if col in trades_df.columns and pd.api.types.is_datetime64_any_dtype(trades_df[col]):
                        if trades_df[col].dt.tz is None:
                            trades_df[col] = trades_df[col].dt.tz_localize('UTC')
                        elif str(trades_df[col].dt.tz) != 'UTC':
                            trades_df[col] = trades_df[col].dt.tz_convert('UTC')
                # Fix model_probabilities column for parquet saving
                if "model_probabilities" in trades_df.columns:
                    def fix_keys(val):
                        if isinstance(val, dict):
                            return {str(k): v for k, v in val.items()}
                        return val
                    trades_df["model_probabilities"] = trades_df["model_probabilities"].apply(fix_keys)
                trades_df.to_parquet(trades_path)
                self.logger.info(f"Trades data saved to {trades_path}")
            else:
                self.logger.info("No trades executed to save.")

            # --- Save Equity Curve ---
            if equity_curve is not None and not equity_curve.empty:
                equity_path = results_dir / equity_pattern.format(**file_params)
                # Ensure index is timezone-aware (UTC)
                if equity_curve.index.tz is None:
                    equity_curve.index = equity_curve.index.tz_localize('UTC')
                elif str(equity_curve.index.tz) != 'UTC':
                    equity_curve.index = equity_curve.index.tz_convert('UTC')
                equity_curve.to_frame(name='equity').to_parquet(equity_path)
                self.logger.info(f"Equity curve data saved to {equity_path}")
            else:
                self.logger.info("Equity curve is empty, not saving.")

        except Exception as e:
            self.logger.error(f"Error saving backtest results: {e}", exc_info=True)

