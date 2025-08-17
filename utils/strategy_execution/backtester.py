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


    def run_backtest(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Executes the backtest simulation over the historical data.

        Returns:
            Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
                - final_trade_history (pd.DataFrame): DataFrame of all completed trades.
                - final_equity_curve (pd.Series): Series representing the account equity over time.
                - summary_metrics (Dict[str, Any]): Dictionary of key performance metrics.
        """
        self.logger.info(f"--- Starting Backtest Simulation for {self.symbol} {self.interval} ({self.model_type}) ---")

        processed_bar_data = None # Initialize to None for the final equity update fallback

        try:
            # Get the data stream generator from MarketDataHandler
            # MarketDataHandler handles loading raw data, feature engineering, and train/test split internally
            data_stream_generator = self.market_data_handler.get_processed_data_stream()

            for i, processed_bar_data in enumerate(data_stream_generator):
                if processed_bar_data is None:
                    self.logger.warning(f"Received None for processed bar data at index {i}. Skipping.")
                    continue

                current_timestamp = processed_bar_data.name
                current_price = processed_bar_data['close']
                signal = processed_bar_data.get('signal', 0)
                model_probabilities = processed_bar_data.get('probabilities')
                if isinstance(model_probabilities, list) and len(model_probabilities) > 0:
                    model_probabilities = pd.Series(model_probabilities[0]) # Convert dict in list to Series
                else:
                    model_probabilities = pd.Series(dtype=float)

                self.logger.debug(f"Processing bar {i} at {current_timestamp}. Signal: {signal}, Price: {current_price:.4f}")

                # --- 1. Manage Open Positions (Check for Exits) ---
                # Check for existing open positions and if any exit conditions are met
                open_trade = self.trading_session_manager.get_open_position()
                if open_trade:
                    # Check 'hard' exit conditions (SL, TP, Liquidation, Max Holding)
                    exit_triggered, exit_reason, exit_price = self.trade_execution_engine.check_exit_conditions(
                        open_trade=open_trade,
                        current_bar_data=processed_bar_data, # Pass the entire bar for OHLC access
                        current_bar_index=i # Pass the current bar index
                    )

                    if exit_triggered:
                        self.logger.info(f"Exit triggered for {open_trade['direction_str']} trade at {exit_price:.{self.app_config.exchange.price_precision}f} due to {exit_reason}.")
                        # Calculate and record the completed trade
                        completed_trade_details = self.trade_execution_engine.calculate_exit_details(
                            open_trade=open_trade,
                            exit_price=exit_price,
                            exit_time=current_timestamp,
                            exit_reason=exit_reason
                        )
                        self.trading_session_manager.close_position(completed_trade_details)
                    else:
                        self.logger.debug(f"Open position held. No hard exit triggered at bar {current_timestamp}.")
                
                # --- 2. Check for Soft Exits (e.g., Neutral Signal) ---
                # This logic comes AFTER hard exits. If a hard exit happens, we prioritize it.
                if open_trade and not exit_triggered:
                    if self.app_config.trading.trade_execution.exit_on_neutral_signal and signal == 0:
                        self.logger.info(f"Neutral signal (0) received for existing {open_trade['direction_str']} trade. Initiating soft exit.")
                        # For simplicity, use current_price as exit price for soft exit
                        completed_trade_details = self.trade_execution_engine.calculate_exit_details(
                            open_trade=open_trade,
                            exit_price=current_price,
                            exit_time=current_timestamp,
                            exit_reason='neutral_signal'
                        )
                        self.trading_session_manager.close_position(completed_trade_details)


                # --- 3. Evaluate New Entry Signal ---
                # Only consider a new entry if no position is currently open AND signal is not neutral.
                if not self.trading_session_manager.get_open_position() and signal != 0:
                    entry_details = self.trade_execution_engine.calculate_entry_details(
                        signal=signal,
                        current_capital=self.trading_session_manager.get_current_equity(),
                        current_price=current_price,
                        current_bar_features=processed_bar_data,
                        model_probabilities=model_probabilities,
                        current_bar_index=i # Pass bar index for detailed logging
                    )

                    if entry_details:
                        self.logger.info(f"Attempting to open {entry_details['direction_str']} position.")
                        self.trading_session_manager.open_position(entry_details)
                    else:
                        self.logger.debug(f"Signal {signal} at {current_timestamp} did not result in an entry (filtered or insufficient funds).")

                # --- 4. Update Equity Curve (End of Bar) ---
                # Always update equity at the end of each bar using the latest known market price
                self.trading_session_manager.update_equity_curve(
                    current_price=current_price,
                    timestamp=current_timestamp
                )

        except Exception as e:
            self.logger.critical(f"An unexpected error occurred during backtest simulation: {e}", exc_info=True)
            # Potentially save partial results here if needed for debugging
        finally:
            self.logger.info("--- Backtest Simulation Finished ---")

            # Finalize equity curve with the last known data point
            self.logger.info("Finalizing backtest results and calculating summary metrics...") # Updated log message
            final_trade_history = self.trading_session_manager.get_trade_history()
            final_equity_curve = self.trading_session_manager.get_equity_curve()
            
            # Ensure last equity point reflects current capital + unrealized PnL (if any open position)
            # Use the last known price to update unrealized PnL
            # This should happen only if there's an open position AND processed_bar_data is available
            if self.trading_session_manager.get_open_position() and processed_bar_data is not None:
                last_price_for_equity = processed_bar_data['close']
                final_equity_value = self.trading_session_manager.get_current_equity(current_price=last_price_for_equity)
                last_timestamp = processed_bar_data.name
                self.trading_session_manager.update_equity_curve(final_equity_value, last_timestamp)
                final_equity_curve = self.trading_session_manager.get_equity_curve() # Re-fetch updated curve
            elif self.trading_session_manager.get_open_position(): # If open position but no processed_bar_data (e.g., error early)
                self.logger.warning("Open position detected at end of backtest but no latest bar data for final equity update. Using current capital.")
                final_equity_value = self.trading_session_manager.get_current_equity() # Gets current capital from manager state
                last_timestamp = final_equity_curve.index[-1] if not final_equity_curve.empty else pd.Timestamp.now(tz='UTC')
                self.trading_session_manager.update_equity_curve(final_equity_value, last_timestamp)
                final_equity_curve = self.trading_session_manager.get_equity_curve()


            # Calculate metrics using PerformanceAnalyzer
            performance_analyzer = PerformanceAnalyzer(
                symbol=self.symbol,
                interval=self.interval,
                model_type=self.model_type,
                initial_capital=self.app_config.trading.risk.initial_capital,
                trade_history_data=final_trade_history,
                equity_data=final_equity_curve,
                results_type='backtest',
                app_config=self.app_config # <-- ADDED THIS LINE!
            )
            # IMPORTANT CHANGE: Only calculate summary metrics here, DO NOT run full analysis (no plotting/saving)
            summary_metrics = performance_analyzer.calculate_summary_metrics()

            return final_trade_history, final_equity_curve, summary_metrics

