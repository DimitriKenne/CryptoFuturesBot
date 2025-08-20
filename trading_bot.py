#!/usr/bin/env python3
# scripts/trading_bot.py

"""
Trading Bot for Futures Markets using a Ternary Classification Model.

Connects to an exchange, fetches data, generates features, gets model signals,
manages positions and orders based on strategy rules, handles errors, and sends notifications.
Leverages modular components for data handling, trade execution, and session management.
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import pandas as pd # For pd.Series in processed_bar_data
from datetime import datetime, timezone
import re # For parsing interval strings

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import main AppConfig and other params
from config.params import AppConfig, app_config, FLOAT_EPSILON
from config.paths import PATHS # For persistent storage paths

# Import modular components with corrected paths
from utils.data_management.market_data_handler import MarketDataHandler
from utils.strategy_execution.trade_execution_engine import TradeExecutionEngine
from utils.strategy_execution.trading_session_manager import TradingSessionManager
from utils.notification_manager import NotificationManager
from utils.exchange_adapters.exchange_interface import ExchangeInterface
from utils.exchange_adapters.binance.futures_adapter import BinanceFuturesAdapter
from utils.exceptions import TemporalSafetyError, ExchangeConnectionError, ConfigurationError, ModelAnalysisError # Import custom exceptions

# Setup for basic console logging. For more advanced logging,
# integrate a dedicated logging utility.
from utils.logger_config import setup_rotating_logging
setup_rotating_logging("trading_bot")
logger = logging.getLogger(__name__)

class TradingBot:
    """
    Orchestrates the live trading process, managing data, signals, trades,
    and session state.
    """

    def __init__(self,
                 app_config: AppConfig,
                 symbol: str,
                 interval: str,
                 model_type: str,
                 notifier: NotificationManager):
        """
        Initializes the TradingBot with necessary components and configurations.

        Args:
            app_config (AppConfig): The comprehensive application configuration object.
            symbol (str): The trading symbol (e.g., "BTCUSDT").
            interval (str): The data interval (e.g., "1h", "5m").
            model_type (str): The type of ML model to use for signals.
            notifier (NotificationManager): The notification manager instance.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing TradingBot...")

        self.app_config = app_config
        self.symbol = symbol.upper()
        self.interval = interval
        self.model_type = model_type
        self.notifier = notifier

        # Initialize core components
        # Exchange Adapter
        self.exchange_adapter: ExchangeInterface = BinanceFuturesAdapter(
            app_config=self.app_config,
            symbol=self.symbol, # Pass symbol
            logger=self.logger, # Pass logger
        )
        # The MarketDataHandler, TradeExecutionEngine, and TradingSessionManager
        # will now be initialized in the `start` async method after the exchange adapter is ready.
        self.market_data_handler = None
        self.trade_execution_engine = None
        self.trading_session_manager = None

        self.last_processed_candle_timestamp: Optional[datetime] = None
        self.last_state_save_time: float = time.time() # Timestamp of the last state save
        self.state_save_interval_seconds: int = 300 # Save state every 5 minutes (300 seconds)

        # Calculate dynamic trade loop interval
        self.trade_loop_interval_seconds = self._calculate_trade_loop_interval()

        self.logger.info(f"TradingBot initialized for {self.symbol} {self.interval} with model {self.model_type}.")
        self.logger.info(f"Bot will check for new data every {self.trade_loop_interval_seconds:.2f} seconds.")

    def _calculate_trade_loop_interval(self) -> float:
        """
        Calculates the appropriate sleep interval for the trading loop dynamically
        based on the candle interval and configured polling frequency.

        Returns:
            float: The calculated sleep interval in seconds.
        """
        interval_value = int(re.findall(r'\d+', self.interval)[0])
        interval_unit = re.findall(r'[a-zA-Z]+', self.interval)[0].lower()

        if interval_unit == 'm':
            candle_duration_minutes = interval_value
        elif interval_unit == 'h':
            candle_duration_minutes = interval_value * 60
        elif interval_unit == 'd':
            candle_duration_minutes = interval_value * 24 * 60
        else:
            self.logger.warning(f"Unsupported interval unit '{interval_unit}'. Defaulting to {self.app_config.general.data_granularity_minutes} minutes.")
            candle_duration_minutes = self.app_config.general.data_granularity_minutes

        candle_duration_seconds = candle_duration_minutes * 60

        # Calculate ideal polling interval based on factor
        base_polling_interval = candle_duration_seconds * self.app_config.general.polling_frequency_factor

        # Ensure the polling interval is at least the minimum configured value
        calculated_interval = max(base_polling_interval, self.app_config.general.min_trade_loop_interval_seconds)

        self.logger.debug(f"Calculated trade loop interval: {calculated_interval:.2f} seconds "
                          f"(Candle: {self.interval}, Duration: {candle_duration_seconds}s, "
                          f"Factor: {self.app_config.general.polling_frequency_factor}, "
                          f"Min: {self.app_config.general.min_trade_loop_interval_seconds}s)")
        return calculated_interval


    async def _load_initial_state(self):
        """
        Loads the bot's state from persistent storage (if applicable) and initializes
        the equity curve.
        """
        self.logger.info("Loading initial trading session state...")
        self.trading_session_manager.load_state(
            symbol=self.symbol,
            interval=self.interval,
            model_type=self.model_type,
            is_live_trading=True
        )

        # Initialize equity curve. If state was loaded, it will start from that point.
        # Otherwise, it starts from initial capital.
        # Need to fetch current time for the initial equity point
        initial_timestamp = datetime.now(timezone.utc)
        self.trading_session_manager.initialize_equity_curve(
            first_bar_time=initial_timestamp,
            initial_capital=self.trading_session_manager.get_current_capital()
        )
        self.logger.info("Initial trading session state loaded/initialized.")


    async def _get_latest_market_data(self) -> Optional[pd.Series]:
        """
        Fetches the latest completed market data bar, processes it through
        feature engineering and model inference, and returns it with the signal.
        """
        self.logger.debug("🔄 Fetching latest processed market data...")
        try:
            processed_bar_data = await self.market_data_handler.get_latest_data(
                exchange_adapter=self.exchange_adapter,
                last_processed_timestamp=self.last_processed_candle_timestamp
            )
            if processed_bar_data is not None:
                self.last_processed_candle_timestamp = processed_bar_data.name
                signal = processed_bar_data.get('signal', 'N/A')
                self.logger.info(f"✅ New bar processed at {self.last_processed_candle_timestamp}. Signal: {signal}")
            return processed_bar_data
        except TemporalSafetyError as e:
            self.logger.error(
                f"❌ Feature safety error: {str(e)}. "
                "Check data quality or adjust feature config."
            )
            await self.notifier.send_notification(
                f"Market Data Error for {self.symbol}-{self.interval}: {str(e)}",
                level='error'
            )
            return None
        except (ExchangeConnectionError, ConfigurationError, ModelAnalysisError, RuntimeError) as e:
            self.logger.error(f"⚠️ Error fetching or processing market data: {e}", exc_info=True)
            await self.notifier.send_notification(f"Market Data Error for {self.symbol}-{self.interval}: {e}", level='error')
            return None
        except Exception as e:
            self.logger.critical(f"❌ Unexpected error in _get_latest_market_data: {e}", exc_info=True)
            await self.notifier.send_notification(f"CRITICAL: Unexpected data error for {self.symbol}-{self.interval}: {e}", level='critical')
            raise

    async def _handle_trading_logic(self, processed_bar_data: Optional[pd.Series]):
        """
        Adds the necessary parameter `trade_execution_engine` and updates the
        calls accordingly.

        Applies trading logic: checks for exit conditions if a position is open,
        or looks for entry signals if no position is open.

        Args:
            processed_bar_data (Optional[pd.Series]): The latest processed market bar data.
        """
        if processed_bar_data is None:
            self.logger.debug("⏸️ No new processed bar data to act on.")
            return

        current_price = processed_bar_data['close']
        current_bar_time = processed_bar_data.name
        
        current_bar_index_for_holding = current_bar_time


        open_position = self.trading_session_manager.get_open_position()
        current_capital = self.trading_session_manager.get_current_capital()
        current_equity = self.trading_session_manager.get_current_equity(current_price)

        # Update equity curve at the current bar's timestamp
        self.trading_session_manager.update_equity_curve(current_equity, current_bar_time)

        # --- Handle Existing Open Position ---
        if open_position:
            self.logger.debug(f"📈 Open {open_position.get('direction_str').upper()} position detected. Checking exit conditions...")
            
            exit_met, exit_reason, exit_price = self.trade_execution_engine.check_exit_conditions(
                open_trade=open_position,
                processed_bar_data=processed_bar_data, # Use processed_bar_data here
                current_bar_index=open_position.get('entry_bar_index', 0) # Use actual entry bar index if available, else 0
            )

            if exit_met:
                self.logger.info(f"🔔 Exit condition met: {exit_reason}. Preparing to close position.")
                
                # If exit_price is NaN (e.g., from invalid_ohlc), use current_price as a fallback
                if pd.isna(exit_price):
                    exit_price = current_price
                    self.logger.warning(f"⚠️ Exit price was NaN, using current_price {exit_price:.{self.app_config.exchange.price_precision}f} as fallback.")

                completed_trade_details = self.trade_execution_engine.calculate_exit_details(
                    open_trade=open_position,
                    exit_price=exit_price,
                    exit_time=current_bar_time,
                    exit_reason=exit_reason,
                    current_bar_index=open_position.get('entry_bar_index', 0) # Re-using entry_bar_index; needs careful handling if integer-based holding.
                )
                
                self.trading_session_manager.close_position(completed_trade_details)
                
                await self.notifier.send_notification(
                    f"TRADE CLOSED: {open_position.get('direction_str').upper()} "
                    f"@ {completed_trade_details.get('exit_price',0):.4f} | "
                    f"Net PnL: {completed_trade_details.get('net_pnl',0.0):.2f} | "
                    f"Reason: {exit_reason} | "
                    f"Current Capital: {self.trading_session_manager.get_current_capital():.2f}",
                    level='info'
                )

        # --- Look for New Entry Signal if No Position Open ---
        else: # No open position
            # Check if trading is allowed for the current volatility regime
            current_regime = processed_bar_data.get(self.app_config.features.volatility_regime_col_name)
            if pd.isna(current_regime):
                current_regime = 0
                self.logger.warning(f"⚠️ Volatility regime is NaN for current bar {current_bar_time}. Defaulting to regime 0.")
            if not self.app_config.trading.volatility_regime.allow_trading.get(int(current_regime), True):
                self.logger.info(f"⏸️ Trading not allowed for current volatility regime {int(current_regime)}. Skipping entry check.")
                return # Skip entry if trading is explicitly disallowed for this regime

            signal = processed_bar_data.get('signal')
            probabilities = processed_bar_data.get('probabilities')

            if signal is not None and signal != 0:
                self.logger.debug(f"📊 Detected signal: {signal} with probabilities: {probabilities}")

                entry_details = self.trade_execution_engine.calculate_entry_details(
                    signal=signal,
                    current_capital=self.trading_session_manager.get_current_capital(),
                    current_price=current_price,
                    current_bar_features=processed_bar_data, # Pass the entire series as features
                    model_probabilities=pd.Series(probabilities) if probabilities else None,
                    current_bar_index=current_bar_time # Using timestamp as a unique identifier for entry bar index
                )

                if entry_details:
                    self.trading_session_manager.set_open_position(entry_details)
                    await self.notifier.send_notification(
                        f"TRADE ENTERED: {entry_details.get('direction_str').upper()} "
                        f"@ {entry_details.get('entry_price',0):.4f} | "
                        f"Qty: {entry_details.get('quantity',0.0):.2f} | "
                        f"SL: {entry_details.get('stop_loss_price',0):.4f} | "
                        f"TP: {entry_details.get('take_profit_price',0):.4f} | "
                        f"Current Capital: {self.trading_session_manager.get_current_capital():.2f}",
                        level='info'
                    )
                else:
                    self.logger.info(f"⏸️ Signal {signal} did not result in an entry (filtered or unable to afford).")
            else:
                self.logger.debug("⏸️ No active signal for entry.")

    async def start(self):
        """
        Performs asynchronous setup for the bot's components that require an active
        exchange connection or other async initialization.
        This method must be called before running the main trade loop.
        """
        self.logger.info("🚀 Performing asynchronous TradingBot setup...")
        # Ensure async setup for exchange adapter is called here before other components
        await self.exchange_adapter.async_setup() # Call async setup for adapter

        # Initialize Market Data Handler (depends on exchange_adapter being ready)
        self.market_data_handler = MarketDataHandler(
            app_config=self.app_config,
            mode='live',
            symbol=self.symbol,
            interval=self.interval,
            model_type=self.model_type
        )

        # Initialize Trade Execution Engine
        self.trade_execution_engine = TradeExecutionEngine(
            app_config=self.app_config
        )

        # Initialize Trading Session Manager
        self.trading_session_manager = TradingSessionManager(
            app_config=self.app_config
        )
        self.logger.info("✅ Asynchronous TradingBot setup complete.")


    async def run(self):
        """
        Main asynchronous loop for the trading bot.
        Continuously fetches data, applies trading logic, and manages state.
        """
        self.logger.info(f"--- Starting Trading Bot Live Run for {self.symbol} {self.interval} ---")
        await self.start()
        await self.notifier.send_notification(f"Trading Bot LIVE for {self.symbol}-{self.interval} started!", level='info')
        try:
            await self._load_initial_state()

            while True:
                start_time = time.time()
                try:
                    # 1. Get latest processed market data
                    processed_bar_data = await self._get_latest_market_data()

                    # 2. Execute trading logic (entry/exit)
                    await self._handle_trading_logic(processed_bar_data)

                    # 3. Periodically save bot state for persistence
                    if (time.time() - self.last_state_save_time) >= self.state_save_interval_seconds:
                        self.trading_session_manager.save_state(
                            symbol=self.symbol,
                            interval=self.interval,
                            model_type=self.model_type,
                            is_live_trading=True
                        )
                        self.last_state_save_time = time.time()
                        self.logger.info("💾 Bot state saved successfully.")
                except asyncio.CancelledError:
                    self.logger.info("🛑 Asyncio task cancelled. Initiating graceful shutdown.")
                    break
                except (ExchangeConnectionError, TemporalSafetyError, ConfigurationError, ModelAnalysisError) as e:
                    self.logger.error(f"⚠️ Recoverable error during main loop: {e}", exc_info=True)
                    await self.notifier.send_notification(f"Bot recoverable error for {self.symbol}-{self.interval}: {e}", level='error')
                except Exception as e:
                    self.logger.critical(f"❌ Critical unhandled error in main loop: {e}", exc_info=True)
                    await self.notifier.send_notification(f"Bot CRITICAL ERROR for {self.symbol}-{self.interval}: {e}", level='critical')
                    raise
                elapsed_time = time.time() - start_time
                sleep_duration = self.trade_loop_interval_seconds - elapsed_time
                if sleep_duration > 0:
                    self.logger.debug(f"⏳ Sleeping for {sleep_duration:.2f} seconds.")
                    await asyncio.sleep(sleep_duration)
                else:
                    self.logger.warning(f"⚠️ Loop took longer than interval ({elapsed_time:.2f}s > {self.trade_loop_interval_seconds:.2f}s). No sleep.")
                    await self.notifier.send_notification(
                        f"Bot performance warning: Loop took {elapsed_time:.2f}s, exceeding interval {self.trade_loop_interval_seconds:.2f}s.",
                        level='warning'
                    )
        except Exception as e:
            self.logger.critical(f"❌ Outer critical error caught in run(): {e}", exc_info=True)
            await self.notifier.send_notification(f"Bot OUTER CRITICAL ERROR: {e}", level='critical')
        finally:
            await self.shutdown()

    async def shutdown(self, critical_error: bool = False):
        """
        Performs a graceful shutdown of the trading bot, saving state and sending notifications.

        Args:
            critical_error (bool): True if shutdown is due to a critical error, False otherwise.
        """
        self.logger.info("🛑 Initiating Trading Bot shutdown...")
        # Save final state before exiting
        try:
            if self.trading_session_manager: # Ensure it's initialized
                self.trading_session_manager.save_state(
                    symbol=self.symbol,
                    interval=self.interval,
                    model_type=self.model_type,
                    is_live_trading=True
                )
                self.logger.info("💾 Final bot state saved.")
        except Exception as e:
            self.logger.error(f"⚠️ Error saving final bot state during shutdown: {e}", exc_info=True)
            await self.notifier.send_notification(f"Error saving final state for {self.symbol}-{self.interval}: {e}", level='error')

        # Send final notification
        if critical_error:
            await self.notifier.send_notification(f"Trading Bot for {self.symbol}-{self.interval} shutting down due to CRITICAL ERROR.", level='critical')
        else:
            await self.notifier.send_notification(f"Trading Bot for {self.symbol}-{self.interval} gracefully shut down.", level='info')

        # Close exchange connection
        try:
            if self.exchange_adapter:
                await self.exchange_adapter.close_connection()
                self.logger.info("🔌 Exchange connection closed.")
        except Exception as e:
            self.logger.error(f"⚠️ Error closing exchange connection: {e}", exc_info=True)
            await self.notifier.send_notification(f"Error closing exchange connection for {self.symbol}-{self.interval}: {e}", level='error')

        self.logger.info("✅ Trading Bot shutdown complete.")


def main():
    """
    Main function to parse arguments and run the trading bot.
    """
    parser = argparse.ArgumentParser(description="Run a live trading bot simulation.")
    parser.add_argument('--symbol', type=str, required=True,
                        help='Trading symbol (e.g., BTCUSDT).')
    parser.add_argument('--interval', type=str, required=True,
                        choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
                        help='Time interval (e.g., 1h, 1d).')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(app_config.model.AVAILABLE_MODEL_TYPES.keys()),
                        help='Model key from app_config.model (e.g., xgboost, lstm).')

    args = parser.parse_args()

    logger.info(f"--- Trading Bot Script Started ({args.symbol} {args.interval} {args.model}) ---")

    notifier_instance: Optional[NotificationManager] = None
    bot_instance: Optional[TradingBot] = None

    try:
        # Initialize Notifier (using the global app_config.notifier parameters)
        # Note: Notifier takes a dict, so pass the relevant part of app_config
        notifier_instance = NotificationManager(config=app_config.notifier.__dict__)

        bot_instance = TradingBot(
            app_config=app_config,
            symbol=args.symbol,
            interval=args.interval,
            model_type=args.model,
            notifier=notifier_instance
        )
        asyncio.run(bot_instance.run()) # The run method now handles calling start() internally

    except (ValueError, ConnectionError, RuntimeError, ExchangeConnectionError, TemporalSafetyError, ConfigurationError, ModelAnalysisError) as e:
        logger.critical(f"Bot failed during initialization or setup: {e}", exc_info=True)
        if notifier_instance:
             asyncio.run(notifier_instance.send_notification(f"Bot CRITICAL STARTUP FAILURE for {args.symbol}-{args.interval}: {e}", level='critical'))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. asyncio.run() will handle graceful shutdown via bot_instance.run().")
        # The finally block in bot_instance.run() should trigger shutdown.
    except Exception as e:
        logger.critical(f"Unexpected critical error during bot execution: {e}", exc_info=True)
        if notifier_instance:
             asyncio.run(notifier_instance.send_notification(f"Bot CRITICAL UNEXPECTED ERROR for {args.symbol}-{args.interval}: {e}", level='critical'))
        if bot_instance:
             logger.info("Attempting emergency shutdown due to unexpected error...")
             try:
                  asyncio.run(bot_instance.shutdown(critical_error=True))
             except Exception as shutdown_err:
                  logger.error(f"Error during emergency shutdown: {shutdown_err}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("--- Trading Bot Script Finished ---")


if __name__ == "__main__":
    main()

