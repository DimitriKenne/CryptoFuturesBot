# utils/strategy_execution/backtester.py

import logging
import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Tuple, Literal

# Add project root to Python path for imports if needed in a standalone run
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Import all modular components
try:
    from config.params import AppConfig
    from utils.data_management.market_data_handler import MarketDataHandler
    from utils.strategy_execution.trade_execution_engine import TradeExecutionEngine
    from utils.strategy_execution.trading_session_manager import TradingSessionManager
except ImportError as e:
    logging.critical(f"Failed to import necessary modules for Backtester: {e}. Ensure all config and utils files are correctly placed.", exc_info=True)
    sys.exit(1)

logger = logging.getLogger(__name__)

class Backtester:
    """
    Orchestrates the backtesting process by simulating trade execution over historical data.
    This class is a pure simulation engine. It does not save files or perform analysis.
    Its sole responsibility is to run the simulation and return the raw results.
    """

    def __init__(self,
                 app_config: AppConfig,
                 symbol: str,
                 interval: str,
                 model_type: str,
                 backtest_mode: Literal['full', 'train', 'test'],
                 train_ratio: float = 0.7,
                 initial_ohlcv_data: Optional[pd.DataFrame] = None # For Monte Carlo
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
            initial_ohlcv_data (Optional[pd.DataFrame]): Optional, pre-loaded OHLCV data.
                                                        If provided, data loading from files is skipped.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initializing Backtester for {symbol}-{interval} with model {model_type} in {backtest_mode} mode.")

        self.app_config = app_config
        self.symbol = symbol.upper()
        self.interval = interval
        self.model_type = model_type

        # Initialize core components
        self.market_data_handler = MarketDataHandler(
            mode='backtest',
            symbol=self.symbol,
            interval=self.interval,
            model_type=self.model_type,
            app_config=self.app_config,
            initial_ohlcv_data=initial_ohlcv_data,
            train_ratio=train_ratio,
            backtest_mode=backtest_mode
        )
        self.trade_execution_engine = TradeExecutionEngine(app_config=self.app_config)
        self.trading_session_manager = TradingSessionManager(app_config=self.app_config)

        self.logger.info("Backtester initialized. Ready to run simulation.")

    def run_backtest(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the backtest simulation over the historical data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - A DataFrame of all completed trades.
                - A DataFrame representing the account equity over time.
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
                    self.trading_session_manager.initialize_equity_curve(current_timestamp)

                if isinstance(model_probabilities, dict):
                    model_probabilities = pd.Series(model_probabilities)
                elif isinstance(model_probabilities, list) and len(model_probabilities) > 0 and isinstance(model_probabilities[0], dict):
                    model_probabilities = pd.Series(model_probabilities[0])
                elif not isinstance(model_probabilities, pd.Series):
                    model_probabilities = pd.Series(dtype=float)

                open_trade = self.trading_session_manager.get_open_position()
                exit_triggered = False

                if open_trade:
                    exit_triggered, exit_reason, exit_price = self.trade_execution_engine.check_exit_conditions(
                        open_trade=open_trade,
                        current_bar_data=processed_bar_data,
                        current_bar_index=i
                    )
                    if exit_triggered:
                        completed_trade_details = self.trade_execution_engine.calculate_exit_details(
                            open_trade=open_trade,
                            exit_price=exit_price,
                            exit_time=current_timestamp,
                            exit_reason=exit_reason,
                            current_bar_index=i
                        )
                        self.trading_session_manager.close_position(completed_trade_details)

                if open_trade and not exit_triggered and self.app_config.trading.trade_execution.exit_on_neutral_signal and signal == 0:
                    completed_trade_details = self.trade_execution_engine.calculate_exit_details(
                        open_trade=open_trade,
                        exit_price=current_price,
                        exit_time=current_timestamp,
                        exit_reason='neutral_signal',
                        current_bar_index=i
                    )
                    self.trading_session_manager.close_position(completed_trade_details)

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
                        self.trading_session_manager.set_open_position(entry_details)

                equity = self.trading_session_manager.get_current_equity(current_price=current_price)
                self.trading_session_manager.update_equity_curve(equity, current_timestamp)
                self.logger.info(f"[BAR {i}] 💰 Equity: {equity:.2f} | Capital: {self.trading_session_manager.get_current_capital():.2f}")

        except Exception as e:
            self.logger.critical(f"Backtest error: {e}", exc_info=True)
        finally:
            self.logger.info(f"\n{'='*40}\n🏁 Backtest Finished\n{'='*40}")

            if self.trading_session_manager.get_open_position() and processed_bar_data is not None:
                open_trade = self.trading_session_manager.get_open_position()
                last_price = processed_bar_data['close']
                last_timestamp = processed_bar_data.name
                self.logger.info(f"Closing open trade at end of backtest: {open_trade['direction_str'].upper()} @ {last_price:.4f}")
                completed_trade_details = self.trade_execution_engine.calculate_exit_details(
                    open_trade=open_trade,
                    exit_price=last_price,
                    exit_time=last_timestamp,
                    exit_reason='end_of_backtest',
                    current_bar_index=i if 'i' in locals() else -1
                )
                self.trading_session_manager.close_position(completed_trade_details)

            final_trades_df = self.trading_session_manager.get_trade_history_df()
            final_equity_df = self.trading_session_manager.get_equity_curve_df()
            
            if not final_equity_df.empty:
                self.logger.info(f"\n{'-'*30}\nFinal Equity: {final_equity_df['equity'].iloc[-1]:.2f}\nTotal Trades: {len(final_trades_df)}\n{'-'*30}")

            return final_trades_df, final_equity_df