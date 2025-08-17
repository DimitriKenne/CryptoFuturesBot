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
from utils.notification_manager import Notifier # Corrected import path
from utils.adapters.exchange_interface import ExchangeInterface # Corrected import path
from utils.adapters.binance_futures_adapter import BinanceFuturesAdapter # Corrected import path

# Import custom exceptions (assuming these are defined in a central exceptions file)
try:
    from utils.exceptions import ExchangeConnectionError, TemporalSafetyError, ConfigurationError # Corrected import path
except ImportError:
    # Define dummy exceptions if not centrally defined, to allow the code to run
    class ExchangeConnectionError(Exception): pass
    class TemporalSafetyError(Exception): pass
    class ConfigurationError(Exception): pass
    logging.warning("Custom exceptions not found at utils.exceptions.exceptions. Using dummy exception classes.")

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, # Default logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main class for the live trading bot. Orchestrates data acquisition,
    signal generation, trade execution, and session management.
    """

    def __init__(self,
                 app_config: AppConfig, # Type hint for AppConfig
                 notifier_instance: Notifier,
                 symbol: str,
                 interval: str,
                 model_type: str):
        """
        Initializes the TradingBot.

        Args:
            app_config (AppConfig): Global application configuration.
            notifier_instance (Notifier): Instance of the notification system.
            symbol (str): Trading symbol (e.g., "BTCUSDT").
            interval (str): Data interval (e.g., "1h").
            model_type (str): Type of ML model used (e.g., "xgboost").
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing TradingBot...")

        self.app_config = app_config
        self.notifier = notifier_instance
        self.symbol = symbol
        self.interval = interval
        self.model_type = model_type

        # Extract core config parameters
        self.trade_loop_interval_seconds = self.app_config.general.trade_loop_interval_seconds
        # self.exit_on_neutral_signal = self.app_config.trading.trade_execution.exit_on_neutral_signal # This is handled by TradeExecutionEngine

        # Initialize core modular components
        self.exchange_adapter: ExchangeInterface = self._initialize_exchange_adapter()
        self.market_data_handler = MarketDataHandler(app_config=self.app_config)
        self.trade_execution_engine = TradeExecutionEngine(app_config=self.app_config)
        self.trading_session_manager = TradingSessionManager(app_config=self.app_config)

        # Load previous session state (capital, open position, history)
        self.trading_session_manager.load_state(
            symbol=self.symbol,
            interval=self.interval,
            model_type=self.model_type,
            is_live_trading=True
        )

        # Track the last processed timestamp to avoid duplicate processing
        # This will be updated by MarketDataHandler when it gets new completed candles
        self.last_processed_bar_timestamp: Optional[datetime] = None
        self.logger.info("TradingBot initialized. Ready to go live.")

    def _initialize_exchange_adapter(self) -> ExchangeInterface:
        """Initializes and returns the appropriate exchange adapter."""
        exchange_type = self.app_config.exchange.exchange.lower()
        if exchange_type == 'binance':
            return BinanceFuturesAdapter(self.app_config.exchange)
        else:
            raise ConfigurationError(f"Unsupported exchange type: {exchange_type}")

    async def _get_signal_and_data(self) -> Optional[pd.Series]:
        """
        Fetches new data, applies feature engineering, and gets the ML signal
        and probabilities using the MarketDataHandler.
        Handles temporal safety and ensures we process only new, completed candles.
        """
        try:
            processed_bar_data = await self.market_data_handler.get_latest_data(
                exchange_adapter=self.exchange_adapter, # Pass the live adapter
                symbol=self.symbol,
                interval=self.interval,
                model_type=self.model_type,
                last_processed_timestamp=self.last_processed_bar_timestamp # Pass to avoid re-processing
            )
            
            # If MarketDataHandler returns None, it means no new completed candle is available
            if processed_bar_data is None:
                self.logger.debug("No new completed candle to process.")
                return None
            
            # Update the last processed timestamp if a new bar was successfully processed
            self.last_processed_bar_timestamp = processed_bar_data.name # The timestamp is the Series index

            return processed_bar_data

        except TemporalSafetyError as e:
            self.logger.critical(f"Temporal safety violation during data acquisition: {e}. Bot cannot continue reliably.", exc_info=True)
            self.notifier.send_critical_alert(f"Bot CRITICAL ERROR: Temporal safety violation for {self.symbol} {self.interval}. {e}")
            raise # Re-raise to trigger graceful shutdown
        except ExchangeConnectionError as e:
            self.logger.error(f"Exchange connection error during data fetch: {e}. Retrying...", exc_info=True)
            self.notifier.send_warning_alert(f"Bot Warning: Exchange connection issue for {self.symbol} {self.interval}. {e}")
            return None # Allow loop to continue and retry
        except Exception as e:
            self.logger.critical(f"Unexpected error during signal and data acquisition: {e}", exc_info=True)
            self.notifier.send_critical_alert(f"Bot CRITICAL ERROR: Data acquisition failed for {self.symbol} {self.interval}. {e}")
            raise # Re-raise to trigger graceful shutdown

    async def _execute_trade_logic(self, processed_bar_data: pd.Series):
        """
        Executes the main trading logic for a given processed bar.
        Handles exit conditions, trade closing, entry conditions, and trade opening.
        """
        timestamp = processed_bar_data.name
        current_price = processed_bar_data['close'] # Use close for general checks/equity update

        open_trade = self.trading_session_manager.get_open_position()
        
        # --- Step 1: Check for Exit Conditions for existing open trade ---
        if open_trade:
            # Check if any exit condition is met
            exit_triggered, exit_reason, exit_price_candidate = \
                self.trade_execution_engine.check_exit_conditions(
                    open_trade=open_trade,
                    current_bar_data=processed_bar_data, # Contains OHLC and filtered signal
                    current_bar_index=open_trade.get('entry_bar_index') # Pass entry_bar_index for holding_bars consistency
                )

            if exit_triggered:
                self.logger.info(f"Exit triggered for {open_trade.get('direction_str')} trade at {exit_price_candidate:.{self.app_config.exchange.price_precision}f} due to {exit_reason} at {timestamp}.")

                # Execute actual exit order on the exchange
                trade_side = 'sell' if open_trade['direction_int'] == 1 else 'buy' # Reverse direction to close
                try:
                    # In a real bot, you'd place a market order here and await its fill details.
                    # For this simulation, we use exit_price_candidate as the filled_price.
                    # You would replace this with actual exchange API calls.

                    # order_result = await self.exchange_adapter.create_market_order(
                    #     symbol=self.symbol,
                    #     side=trade_side,
                    #     quantity=open_trade['quantity']
                    # )
                    # filled_price = order_result.get('avgPrice', exit_price_candidate)
                    # filled_quantity = order_result.get('executedQty', open_trade['quantity'])
                    # exit_order_id = order_result.get('orderId')

                    filled_price = exit_price_candidate
                    filled_quantity = open_trade['quantity']
                    exit_order_id = f"simulated_exit_{int(time.time())}" # Dummy ID

                    if filled_quantity < open_trade['quantity'] * (1 - FLOAT_EPSILON):
                        self.logger.warning(f"Partial fill on exit! Requested {open_trade['quantity']:.4f}, Filled {filled_quantity:.4f}. This requires more complex position management.")
                        self.notifier.send_warning_alert(f"Bot Warning: Partial fill on exit for {self.symbol}. Manual intervention might be needed.")
                        # In a real bot, handle partial fills (e.g., reduce open_position quantity, retry closing remainder)
                        # For this example, we proceed with the filled quantity for PnL calculation

                    # Calculate comprehensive exit details (PnL, fees) with actual filled price
                    completed_trade = self.trade_execution_engine.calculate_exit_details(
                        open_trade=open_trade,
                        exit_price=filled_price, # Use the actual filled price
                        exit_time=timestamp, # Use current bar's timestamp as exit_time
                        exit_reason=exit_reason,
                        current_bar_index=open_trade.get('entry_bar_index') # Re-use entry_bar_index for consistency if needed for holding_bars
                    )
                    # Add exit order ID to completed trade if available
                    completed_trade['exit_order_id'] = exit_order_id


                    # Add completed trade to history and update capital in session manager
                    self.trading_session_manager.add_completed_trade(completed_trade)
                    self.trading_session_manager.clear_open_position()
                    
                    self.logger.info(f"Position closed. New capital: {self.trading_session_manager.get_current_capital():.2f}")
                    self.notifier.send_info_alert(f"Trade closed for {self.symbol} ({exit_reason}). Net PnL: {completed_trade.get('net_pnl'):.2f}. Capital: {self.trading_session_manager.get_current_capital():.2f}")
                    
                    # After closing, if exit was due to reversal, allow for new entry in the same bar cycle
                    if exit_reason == 'reversal_signal':
                        self.logger.info(f"Reversal exit detected. Attempting immediate re-entry if signal valid.")
                        # This allows the entry logic to run immediately after a reversal exit
                        # ensuring a direct flip if the new signal is strong.
                    else:
                        # For other exits, we are now flat and will evaluate entry on next bar's signal
                        return # Exit function, no new entry in this cycle for non-reversal exits

                except ExchangeConnectionError as e: # Specific error for exchange issues
                    self.logger.error(f"Exchange connection error during exit order placement: {e}. Position might still be open!", exc_info=True)
                    self.notifier.send_critical_alert(f"Bot CRITICAL ERROR: Exit order failed for {self.symbol}. Position might be stuck! {e}")
                    # Re-raise to halt the bot if we can't exit
                    raise ExchangeConnectionError(f"Failed to close position: {e}") from e
                except Exception as e: # General fallback
                    self.logger.critical(f"Unexpected error during exit order placement: {e}. Position might still be open!", exc_info=True)
                    self.notifier.send_critical_alert(f"Bot CRITICAL ERROR: Unexpected error during exit for {self.symbol}. Position might be stuck! {e}")
                    # Re-raise to halt the bot
                    raise # Re-raise to trigger graceful shutdown


        # --- Step 2: Check for Entry Conditions ---
        # Only attempt a new entry if no position is open OR if a reversal just happened
        # and we passed through the reversal exit logic in this cycle.
        if not self.trading_session_manager.get_open_position():
            current_filtered_signal = processed_bar_data.get('signal')
            current_probabilities = processed_bar_data.get('probabilities')
            current_capital = self.trading_session_manager.get_current_capital()

            if current_filtered_signal is not None and current_filtered_signal != 0:
                # Use current_price from this bar for calculate_entry_details
                entry_details = self.trade_execution_engine.calculate_entry_details(
                    signal=current_filtered_signal,
                    current_capital=current_capital,
                    current_price=processed_bar_data['close'], # For live, typically use close for signal bar entry
                    current_bar_features=processed_bar_data,
                    model_probabilities=current_probabilities,
                    current_bar_index=None # No sequential int index in live, manage with timestamp
                )

                if entry_details:
                    self.logger.info(f"Attempting to open {entry_details.get('direction_str')} trade of {entry_details.get('quantity'):.{self.app_config.exchange.quantity_precision}f} at {entry_details.get('entry_price'):.{self.app_config.exchange.price_precision}f}.")
                    
                    trade_side = 'buy' if entry_details['direction_int'] == 1 else 'sell'
                    try:
                        # Place actual market order on the exchange
                        # For actual live trading, you'd call exchange_adapter.create_market_order()
                        # and then poll for its fill details.
                        # For simplicity in this structure, we simulate immediate fill at entry_details['entry_price']
                        # In a real bot, fetch actual fill price.

                        # order_result = await self.exchange_adapter.create_market_order(
                        #     symbol=self.symbol,
                        #     side=trade_side,
                        #     quantity=entry_details['quantity']
                        # )
                        # filled_price = order_result.get('avgPrice', entry_details['entry_price'])
                        # filled_quantity = order_result.get('executedQty', entry_details['quantity'])
                        # entry_order_id = order_result.get('orderId')

                        filled_price = entry_details['entry_price']
                        filled_quantity = entry_details['quantity']
                        entry_order_id = f"simulated_entry_{int(time.time())}" # Dummy ID

                        # Update entry_details with actual filled price/quantity from exchange
                        entry_details['entry_price'] = filled_price
                        entry_details['quantity'] = filled_quantity
                        entry_details['entry_order_id'] = entry_order_id

                        self.trading_session_manager.set_open_position(entry_details)
                        self.logger.info(f"Position opened: {entry_details.get('direction_str')} {filled_quantity:.{self.app_config.exchange.quantity_precision}f} @ {filled_price:.{self.app_config.exchange.price_precision}f} at {timestamp}.")
                        self.notifier.send_info_alert(f"Trade opened for {self.symbol} ({entry_details.get('direction_str')}). Qty: {filled_quantity:.2f}, Price: {filled_price:.2f}. Capital: {self.trading_session_manager.get_current_capital():.2f}")

                    except ExchangeConnectionError as e:
                        self.logger.error(f"Exchange connection error during order placement: {e}. Trade not opened.", exc_info=True)
                        self.notifier.send_warning_alert(f"Bot Warning: Order placement failed for {self.symbol}. {e}")
                    except Exception as e:
                        self.logger.critical(f"Unexpected error during order placement for entry: {e}. Trade not opened.", exc_info=True)
                        self.notifier.send_critical_alert(f"Bot CRITICAL ERROR: Order placement failed for {self.symbol}. {e}")
                        raise # Re-raise to trigger graceful shutdown
                else:
                    self.logger.debug(f"Entry not generated for signal {current_filtered_signal} (TradeExecutionEngine filters) at {timestamp}.")
            else:
                self.logger.debug(f"No valid entry signal ({current_filtered_signal}) for bar at {timestamp}.")


    async def run(self):
        """
        Runs the main asynchronous trading bot loop.
        Continuously fetches data, executes trade logic, and manages session state.
        """
        self.logger.info(f"--- Starting Live Trading Bot for {self.symbol} {self.interval} ({self.model_type}) ---")
        self.notifier.send_info_alert(f"Live Trading Bot started for {self.symbol} {self.interval} ({self.model_type}). Initial capital: {self.trading_session_manager.get_current_capital():.2f}")

        try:
            while True:
                start_time = time.time()
                self.logger.debug(f"Fetching and processing data at {datetime.now(timezone.utc)}...")

                # 1. Fetch data, generate features, and get ML signal/probabilities
                processed_bar_data = await self._get_signal_and_data()

                if processed_bar_data is not None: # Only proceed if a new completed bar was processed
                    self.logger.debug(f"Processing new bar: {processed_bar_data.name}. Signal: {processed_bar_data.get('signal')}")

                    # 2. Execute trading logic (check exits, then entries)
                    await self._execute_trade_logic(processed_bar_data)

                    # 3. Update session state (equity curve, periodic saving)
                    current_equity = self.trading_session_manager.get_current_equity(current_price=processed_bar_data['close'])
                    self.trading_session_manager.update_equity_curve(
                        current_equity=current_equity,
                        timestamp=processed_bar_data.name # Use the bar's timestamp
                    )
                    # Save state periodically
                    self.trading_session_manager.save_state(
                        symbol=self.symbol,
                        interval=self.interval,
                        model_type=self.model_type,
                        is_live_trading=True
                    )
                else:
                    self.logger.debug("No new bar to process in this cycle.")

                # Calculate sleep time to maintain desired loop interval
                end_time = time.time()
                time_elapsed = end_time - start_time
                sleep_duration = max(0, self.trade_loop_interval_seconds - time_elapsed)
                self.logger.debug(f"Cycle completed in {time_elapsed:.2f}s. Sleeping for {sleep_duration:.2f}s.")
                await asyncio.sleep(sleep_duration)

        except (KeyboardInterrupt, asyncio.CancelledError):
            self.logger.info("Bot received shutdown signal (KeyboardInterrupt/CancelledError). Initiating graceful shutdown.")
            await self.shutdown()
        except Exception as e:
            self.logger.critical(f"Critical unexpected error in main bot loop: {e}", exc_info=True)
            self.notifier.send_critical_alert(f"Bot CRITICAL ERROR: Unexpected error in main loop. {e}")
            await self.shutdown(critical_error=True) # Pass flag to indicate critical shutdown

    async def shutdown(self, critical_error: bool = False):
        """
        Handles graceful shutdown of the trading bot.
        Closes open positions (if any), saves final state, and cleans up.

        Args:
            critical_error (bool): True if shutdown is due to a critical error.
        """
        self.logger.info("--- Initiating Trading Bot Shutdown ---")
        if not critical_error:
            self.notifier.send_info_alert(f"Trading Bot for {self.symbol} {self.interval} ({self.model_type}) is shutting down gracefully.")
        else:
            self.notifier.send_critical_alert(f"Trading Bot for {self.symbol} {self.interval} ({self.model_type}) is shutting down due to CRITICAL ERROR.")

        try:
            # 1. Close any open positions (optional, depending on strategy)
            # In a real bot, you might attempt to close market orders here.
            open_position = self.trading_session_manager.get_open_position()
            if open_position:
                self.logger.warning(f"Bot is shutting down with an open position: {open_position.get('direction_str')} {open_position.get('quantity'):.4f}.")
                self.notifier.send_warning_alert(f"Bot Warning: Shutting down with open position for {self.symbol}. Please close manually if required.")
                # Attempt to close the position if desired, or leave it for manual handling
                # try:
                #     trade_side = 'sell' if open_position['direction_int'] == 1 else 'buy'
                #     await self.exchange_adapter.create_market_order(
                #         symbol=self.symbol,
                #         side=trade_side,
                #         quantity=open_position['quantity']
                #     )
                #     self.logger.info("Attempted to close open position during shutdown.")
                # except Exception as e:
                #     self.logger.error(f"Failed to close open position during shutdown: {e}")
            
            # 2. Save final session state
            self.trading_session_manager.save_state(
                symbol=self.symbol,
                interval=self.interval,
                model_type=self.model_type,
                is_live_trading=True
            )
            self.logger.info("Final session state saved.")

            # 3. Log final performance summary (optional, or done by a separate reporting tool)
            # For live trading, you might just log capital, or save detailed metrics if desired.
            final_capital = self.trading_session_manager.get_current_capital()
            self.logger.info(f"Final Capital at Shutdown: {final_capital:.2f}")

            # You could optionally run a PerformanceAnalyzer here to generate a summary report
            # performance_analyzer = PerformanceAnalyzer(...)
            # performance_analyzer.run_analysis()

        except Exception as e:
            self.logger.error(f"Error during graceful shutdown process: {e}", exc_info=True)
            self.notifier.send_critical_alert(f"Bot CRITICAL ERROR: Error during shutdown for {self.symbol}. {e}")
        finally:
            self.logger.info("--- Trading Bot Shutdown Complete ---")
            logging.shutdown()
            if critical_error:
                sys.exit(1) # Exit with error code if it was a critical shutdown


def main():
    """
    Main function to parse arguments and start the trading bot.
    """
    parser = argparse.ArgumentParser(description="Run the live trading bot.")
    parser.add_argument('--symbol', type=str, required=True,
                        help='Trading symbol (e.g., BTCUSDT).')
    parser.add_argument('--interval', type=str, required=True,
                        help='Data interval (e.g., 1h, 5m).')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(app_config.model.AVAILABLE_MODEL_TYPES.keys()),
                        help=f"Model type to use for signals. Available: {list(app_config.model.AVAILABLE_MODEL_TYPES.keys())}")

    args = parser.parse_args()

    # Initialize notifier globally here before bot instance
    # Ensure Notifier is imported correctly based on new path
    notifier_instance = Notifier(app_config.notifier)

    logger.info("--- Starting Trading Bot Script ---")
    
    bot_instance = None
    try:
        bot_instance = TradingBot(
            app_config=app_config,
            notifier_instance=notifier_instance,
            symbol=args.symbol,
            interval=args.interval,
            model_type=args.model
        )
        asyncio.run(bot_instance.run())

    except (ValueError, ConnectionError, RuntimeError, ExchangeConnectionError, TemporalSafetyError, ConfigurationError) as e:
        logger.critical(f"Bot failed during initialization or setup: {e}", exc_info=True)
        if notifier_instance:
             notifier_instance.send_critical_alert(f"Bot CRITICAL STARTUP FAILURE: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. asyncio.run() will handle graceful shutdown via bot_instance.run().")
        # The finally block in bot_instance.run() should trigger shutdown.
    except Exception as e:
        logger.critical(f"Unexpected critical error during bot execution: {e}", exc_info=True)
        if notifier_instance:
             notifier_instance.send_critical_alert(f"Bot CRITICAL UNEXPECTED ERROR: {e}")
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

