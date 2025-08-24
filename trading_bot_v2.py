# trading_bot.py

import asyncio
import logging
import signal
import argparse
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Configuration and Core Components ---
from config.params import app_config, AppConfig
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

    def __init__(self, config: AppConfig, symbol: str, interval: str, model_type: str, notifier: NotificationManager):
        self.config = config
        self.symbol = symbol
        self.interval = interval
        self.model_type = model_type
        self.notifier = notifier

        self.is_running = False
        self.shutdown_requested = False

        logger.info("Initializing bot components...")
        data_manager = DataManager()
        exchange_adapter = BinanceFuturesAdapter(app_config=config, symbol=self.symbol, logger=logging.getLogger(BinanceFuturesAdapter.__name__))
        market_data_handler = MarketDataHandler(app_config=config, mode='live', symbol=self.symbol, interval=self.interval, model_type=self.model_type)
        session_manager = LiveTradingSessionManager(app_config=config)
        trade_execution_engine = TradeExecutionEngine(app_config=config, exchange_adapter=exchange_adapter)
        
        self.lifecycle_manager = LifecycleManager(
            data_manager, exchange_adapter, session_manager,
            config.general.bot_id,  # Pass bot_id from the correct config location
            self.model_type, self.symbol, self.interval
        )
        self.trade_cycle_processor = TradeCycleProcessor(
            data_manager, market_data_handler, exchange_adapter, 
            session_manager, trade_execution_engine, self.notifier,
            config.general.bot_id,  # Pass bot_id from the correct config location
            self.model_type, self.symbol, self.interval
        )
        logger.info(f"All bot components initialized with bot_id: '{config.general.bot_id}'")

    async def run(self):
        """Starts the bot, runs the main trading loop, and handles shutdown."""
        self.logger.info(f"--- Starting Trading Bot Live Run for {self.symbol} {self.interval} ---")
        await self.notifier.send_notification(f"Trading Bot LIVE for {self.symbol}-{self.interval} starting up...", level='info')

        try:
            last_processed_timestamp = await self.lifecycle_manager.startup()
            self.trade_cycle_processor.set_last_processed_timestamp(last_processed_timestamp)
            self.is_running = True
            
            loop_interval = self.config.general.min_trade_loop_interval_seconds
            self.logger.info(f"Bot is running. Main loop will cycle every {loop_interval} seconds.")
            await self.notifier.send_notification(f"Trading Bot for {self.symbol}-{self.interval} is now fully operational.", level='info')

            while self.is_running:
                try:
                    await self.trade_cycle_processor.process_cycle()
                    await asyncio.sleep(loop_interval)
                except asyncio.CancelledError:
                    self.logger.info("Asyncio task cancelled. Initiating graceful shutdown.")
                    self.request_shutdown()
                except ExchangeConnectionError as e:
                    self.logger.error(f"A recoverable exchange error occurred: {e}. Continuing...", exc_info=True)
                    await self.notifier.send_notification(f"Bot recoverable error for {self.symbol}-{self.interval}: {e}", level='error')
                    await asyncio.sleep(loop_interval * 2)
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
        await self.lifecycle_manager.shutdown(self.trade_cycle_processor)
        await self.notifier.send_notification(f"Trading Bot for {self.symbol}-{self.interval} has shut down.", level='info')
        self.logger.info("✅ Trading Bot shutdown complete.")

def main():
    """Main function to parse arguments and run the trading bot."""
    parser = argparse.ArgumentParser(description="Run a live trading bot.")
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol (e.g., BTCUSDT).')
    parser.add_argument('--interval', type=str, required=True, choices=['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'], help='Time interval (e.g., 1h, 1d).')
    parser.add_argument('--model_type', type=str, required=True, choices=list(app_config.model.AVAILABLE_MODEL_TYPES.keys()), help='Model key from app_config.model (e.g., xgboost, lstm).')
    args = parser.parse_args()

    logger.info(f"--- Trading Bot Script Started ({args.symbol} {args.interval} {args.model_type}) ---")

    try:
        validate_config(app_config)
    except (ValueError, TypeError) as e:
        logger.critical(f"Invalid application configuration: {e}", exc_info=True)
        sys.exit(1)

    notifier = NotificationManager(config=app_config.notifier.__dict__)
    bot = TradingBot(
        config=app_config,
        symbol=args.symbol,
        interval=args.interval,
        model_type=args.model,
        notifier=notifier
    )

    def handle_shutdown_signal(sig, frame):
        logger.warning(f"Signal {sig} received. Initiating graceful shutdown...")
        bot.request_shutdown()

    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Graceful shutdown should be handled by the signal handler.")
    except Exception as e:
        logger.critical(f"A top-level unhandled exception occurred: {e}", exc_info=True)
        asyncio.run(notifier.send_notification(f"Bot CRITICAL UNEXPECTED SHUTDOWN for {args.symbol}-{args.interval}: {e}", level='critical'))
    finally:
        logger.info("--- Trading Bot Script Finished ---")

if __name__ == "__main__":
    main()