import asyncio
from html import parser
import logging
import signal
import argparse
import sys
import re
from pathlib import Path
import json

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Configuration and Core Components ---
from config.params import app_config, AppConfig
from config.trading import TradingConfig
from config.validator import validate_config
from utils.data_management.data_manager import DataManager
from utils.exchange_adapters.binance.futures_adapter import BinanceFuturesAdapter
from utils.data_management.market_data_handler import MarketDataHandler
from utils.strategy_execution.live_trading_session_manager import LiveTradingSessionManager
from utils.strategy_execution.trade_execution_engine import TradeExecutionEngine
from utils.bot_management.lifecycle_manager import LifecycleManager
from utils.bot_management.trade_cycle_processor import TradeCycleProcessor
from utils.exceptions import ExchangeConnectionError, ConfigurationError
from utils.notification_manager import NotificationManager
from utils.logger_config import setup_rotating_logging

# --- Setup Logger ---
setup_rotating_logging("trading_bot")
logger = logging.getLogger(__name__)

class TradingBot:
    """The main orchestrator for the live trading bot."""

    def __init__(self, config: AppConfig, symbol: str, interval: str, model_type: str, mode: str, notifier: NotificationManager):
        self.logger = logging.getLogger(self.__class__.__name__)    
        self.config = config
        self.symbol = symbol
        self.interval = interval
        self.model_type = model_type
        self.mode = mode
        self.notifier = notifier

        self.is_running = False
        self.shutdown_requested = False
        self.stop_event = asyncio.Event()

        logger.info("Initializing bot components...")
        data_manager = DataManager()
        exchange_adapter = BinanceFuturesAdapter(app_config=config, symbol=self.symbol, logger=logging.getLogger(BinanceFuturesAdapter.__name__))
        market_data_handler = MarketDataHandler(app_config=config, mode='live', symbol=self.symbol, interval=self.interval, model_type=self.model_type)
        session_manager = LiveTradingSessionManager(app_config=config)
        trade_execution_engine = TradeExecutionEngine(app_config=config, exchange_adapter=exchange_adapter, symbol=self.symbol)

        self.trade_cycle_processor = TradeCycleProcessor(
            data_manager, market_data_handler, exchange_adapter, 
            session_manager, trade_execution_engine, self.notifier,
            self.model_type, self.symbol, self.interval, self.mode
        )
        self.lifecycle_manager = LifecycleManager(
            data_manager, exchange_adapter, session_manager,
            self.model_type, self.symbol, self.interval, self.trade_cycle_processor, self.notifier
        )

        logger.info("All bot components initialized.")

        # Calculate dynamic trade loop interval using polling_frequency_factor
        self.trade_loop_interval_seconds = self._calculate_trade_loop_interval()
        logger.info(f"Bot will check for new data every {self.trade_loop_interval_seconds:.2f} seconds (interval: {self.interval}, factor: {self.config.general.polling_frequency_factor}).")

    def _calculate_trade_loop_interval(self) -> float:
        """
        Calculates the appropriate sleep interval for the trading loop dynamically
        based on the candle interval and configured polling frequency.
        """
        # Parse interval (e.g. '5m', '1h', '1d')
        match = re.match(r'^(\d+)([a-zA-Z]+)$', self.interval)
        if match:
            interval_value = int(match.group(1))
            interval_unit = match.group(2).lower()
        else:
            self.logger.warning(f"Could not parse interval '{self.interval}'. Defaulting to {self.config.general.data_granularity_minutes}m.")
            interval_value = self.config.general.data_granularity_minutes
            interval_unit = 'm'

        if interval_unit == 'm':
            candle_duration_minutes = interval_value
        elif interval_unit == 'h':
            candle_duration_minutes = interval_value * 60
        elif interval_unit == 'd':
            candle_duration_minutes = interval_value * 24 * 60
        elif interval_unit == 'w':
            candle_duration_minutes = interval_value * 7 * 24 * 60
        elif interval_unit == 'M':
            candle_duration_minutes = interval_value * 30 * 24 * 60
        else:
            self.logger.warning(f"Unsupported interval unit '{interval_unit}'. Defaulting to {self.config.general.data_granularity_minutes} minutes.")
            candle_duration_minutes = self.config.general.data_granularity_minutes

        candle_duration_seconds = candle_duration_minutes * 60

        # Calculate polling interval
        base_polling_interval = candle_duration_seconds * self.config.general.polling_frequency_factor
        min_interval = self.config.general.min_trade_loop_interval_seconds

        trade_loop_interval = max(base_polling_interval, min_interval)
        self.logger.debug(f"Calculated trade loop interval: {trade_loop_interval:.2f} seconds "
                          f"(Candle: {self.interval}, Duration: {candle_duration_seconds}s, "
                          f"Factor: {self.config.general.polling_frequency_factor}, "
                          f"Min: {self.config.general.min_trade_loop_interval_seconds}s)")
        return trade_loop_interval

    async def run(self):
        """Starts the bot, runs the main trading loop, and handles shutdown."""
        self.logger.info(f"--- Starting Trading Bot Live Run for {self.symbol} {self.interval} ---")
        await self.notifier.send_notification(f"Trading Bot LIVE for {self.symbol}-{self.interval} starting up...", level='info')

        try:
            last_processed_timestamp = await self.lifecycle_manager.startup()
            self.trade_cycle_processor.set_last_processed_timestamp(last_processed_timestamp)
            self.is_running = True
            
            self.logger.info(f"Bot is running. Main loop will cycle every {self.trade_loop_interval_seconds:.2f} seconds.")
            await self.notifier.send_notification(f"Trading Bot for {self.symbol}-{self.interval} is now fully operational.", level='info')

            while self.is_running:
                start_time = asyncio.get_event_loop().time()
                try:
                    await self.trade_cycle_processor.process_cycle()
                    elapsed_time = asyncio.get_event_loop().time() - start_time
                    sleep_duration = self.trade_loop_interval_seconds - elapsed_time
                    if sleep_duration > 0:
                        self.logger.debug(f"Sleeping for {sleep_duration:.2f} seconds.")
                        # await asyncio.sleep(sleep_duration)
                        try:
                            await asyncio.wait_for(self.stop_event.wait(), timeout=sleep_duration)
                        except asyncio.TimeoutError:
                            pass
                    else:
                        self.logger.warning(
                            f"Loop took longer than interval ({elapsed_time:.2f}s > {self.trade_loop_interval_seconds:.2f}s). No sleep."
                        )
                        await self.notifier.send_notification(
                            f"Bot performance warning: Loop took {elapsed_time:.2f}s, exceeding interval {self.trade_loop_interval_seconds:.2f}s.",
                            level='warning'
                        )
                except asyncio.CancelledError:
                    self.logger.info("Asyncio task cancelled. Initiating graceful shutdown.")
                    self.request_shutdown()
                except ExchangeConnectionError as e:
                    self.logger.error(f"A recoverable exchange error occurred: {e}. Continuing...", exc_info=True)
                    await self.notifier.send_notification(f"Bot recoverable error for {self.symbol}-{self.interval}: {e}", level='error')
                    await asyncio.sleep(self.trade_loop_interval_seconds * 2)
                except Exception as e:
                    self.logger.critical("A critical unexpected error occurred in the main loop. Shutting down...", exc_info=True)
                    await self.notifier.send_notification(f"Bot CRITICAL ERROR for {self.symbol}-{self.interval}: {e}", level='critical')
                    self.request_shutdown()
        
        except (ExchangeConnectionError, ConfigurationError) as e:
            self.logger.critical(f"Bot failed to start due to a critical error: {e}", exc_info=True)
            await self.notifier.send_notification(f"Bot CRITICAL STARTUP FAILURE for {self.symbol}-{self.interval}: {e}", level='critical')
        
        finally:
            if self.is_running:
                self.request_shutdown()
            await self.shutdown()

    def request_shutdown(self):
        """Flags that a shutdown has been requested."""
        if not self.shutdown_requested:
            self.logger.info("Shutdown requested. The bot will stop after the current cycle.")
            self.shutdown_requested = True
            self.is_running = False

    async def shutdown(self):
        """Performs a graceful shutdown of the trading bot."""
        self.logger.info("🛑 Initiating Trading Bot shutdown...")
        await self.lifecycle_manager.shutdown()
        await self.notifier.send_notification(f"Trading Bot for {self.symbol}-{self.interval} has shut down.", level='info')
        self.logger.info("✅ Trading Bot shutdown complete.")

def main():
    """Main function to parse arguments and run the trading bot."""
    parser = argparse.ArgumentParser(description="Run a live trading bot.")
    parser.add_argument('--symbol', type=str, required=True, default= app_config.trading.symbol, help='Trading symbol (e.g., BTCUSDT).')
    parser.add_argument('--interval', type=str, required=True, choices=[
        '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
    ], default=app_config.trading.interval, help='Time interval (e.g., 1h, 1d).')
    parser.add_argument('--model_type', type=str, required=True, choices=list(app_config.model.AVAILABLE_MODEL_TYPES.keys()), default=app_config.trading.model_type, help='Model key from app_config.model (e.g., xgboost, lstm).')
    parser.add_argument('--mode', type=str, choices=['automatic', 'hybrid'], default='automatic', help='Trading bot mode: automatic or hybrid.')
        # --- Configuration File Argument ---
    parser.add_argument(
        "--config_file",
        type=str,
        help=(
            "The path to a JSON file to override the default configuration.\n"
            "This file must contain a valid JSON object."
        )
    )

    args = parser.parse_args()
    
    logger.info(f"--- Trading Bot Script Started ({args.symbol} {args.interval} {args.model_type}) ---")

    # --- Load and Apply Configuration ---
    current_app_config = app_config  # Start with the default global config

    if args.config_file:
        try:
            config_file_path = Path(args.config_file)
            if not config_file_path.is_file():
                raise FileNotFoundError(f"Configuration file not found at: {config_file_path}")
            
            with open(config_file_path, 'r') as f:
                override_data = json.load(f)
            
            # Use AppConfig's __init__ to merge the configurations
            current_app_config = AppConfig(**{
                **current_app_config.__dict__,
                **override_data,
            })
            logger.info(f"Configuration overridden with JSON file from: {args.config_file}")
        except FileNotFoundError as e:
            logger.critical(e, exc_info=True)
            sys.exit(1)
        except json.JSONDecodeError as e:
            logger.critical(f"Invalid JSON format in file {args.config_file}: {e}", exc_info=True)
            sys.exit(1)

    # Validate the final configuration
    try:
        validate_config(current_app_config)
    except (ValueError, TypeError) as e:
        logger.critical(f"Invalid application configuration: {e}", exc_info=True)
        sys.exit(1)

    notifier = NotificationManager(config=current_app_config.notifier.__dict__)
    bot = TradingBot(
        config=current_app_config,
        symbol=args.symbol,
        interval=args.interval,
        model_type=args.model_type,
        mode=args.mode,
        notifier=notifier,
    )

    def handle_shutdown_signal(sig, frame):
        logger.warning(f"Signal {sig} received. Initiating graceful shutdown...")
        bot.request_shutdown()
        bot.stop_event.set()  # <-- Wake up main loop immediately

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Graceful shutdown should be handled by the signal handler.")
    except Exception as e:
        logger.critical(f"A top-level unhandled exception occurred: {e}", exc_info=True)
        # Try to send notification while loop is alive
        try:
            asyncio.run(notifier.send_notification(f"Bot CRITICAL UNEXPECTED SHUTDOWN for {args.symbol}-{args.interval}: {e}", level='critical'))
        except Exception as notify_err:
            logger.error(f"Failed to send error notification (loop may be closed): {notify_err}")
    finally:
        logger.info("--- Trading Bot Script Finished ---")


if __name__ == "__main__":
    main()