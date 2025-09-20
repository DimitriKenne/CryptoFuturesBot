import logging
from typing import TYPE_CHECKING, Optional
import uuid

from config.params import app_config
from utils.notification_manager import NotificationManager
from utils.data_management.data_manager import DataManager
from utils.exchange_adapters.exchange_interface import ExchangeInterface
from utils.strategy_execution.live_trading_session_manager import LiveTradingSessionManager
from utils.exceptions import ExchangeConnectionError, ConfigurationError

if TYPE_CHECKING:
    from .trade_cycle_processor import TradeCycleProcessor

logger = logging.getLogger(__name__)

class LifecycleManager:
    """
    Manages the startup and shutdown phases of the trading bot, including robust state reconciliation.
    Handles position/order adoption, orphan/ghost cleanup, and SL/TP recreation.
    """

    def __init__(
        self,
        data_manager: DataManager,
        exchange_adapter: ExchangeInterface,
        session_manager: LiveTradingSessionManager,
        model_type: str,
        symbol: str,
        interval: str,
        trade_cycle_processor: 'TradeCycleProcessor',
        notifier: NotificationManager
    ):
        self.data_manager = data_manager
        self.exchange_adapter = exchange_adapter
        self.session_manager = session_manager
        self.model_type = model_type
        self.symbol = symbol
        self.interval = interval
        self.trade_cycle_processor = trade_cycle_processor
        self.notifier = notifier

    async def startup(self):
        """
        Handles the complete startup procedure: loading state, connecting, and robustly synchronizing.
        This method is the core of the reconciliation logic, checking for a saved state,
        comparing it to the live exchange state, and correcting any inconsistencies before starting the
        main trading loop.
        """
        logger.info("--- Bot Lifecycle: Startup Phase ---")
        
        # Step 0: Connect to the exchange
        await self.exchange_adapter.async_setup()
        
        # 1. Load bot's internal state from the database
        bot_state = self.data_manager.load_bot_state(self.model_type, self.symbol, self.interval)
        if not bot_state:
            logger.warning("No saved bot state found. Starting fresh.")
            bot_state = {}
        bot_position = bot_state.get("open_position")
        
        # 2. Fetch live state from the exchange
        try:
            exchange_open_positions = await self.exchange_adapter.get_open_positions(self.symbol)
            exchange_open_orders = await self.exchange_adapter.get_open_orders(self.symbol)
        except ExchangeConnectionError as e:
            logger.critical(f"Failed to connect to exchange during startup: {e}. Cannot proceed.")
            await self.notifier.send_notification(f"Bot failed to start: Exchange connection error.", level='critical')
            raise

        logger.info(f"Bot state position: {bool(bot_position)}. Exchange position count: {len(exchange_open_positions)}. Exchange orders count: {len(exchange_open_orders)}")

        # --- Reconcile States (The Scenarios) ---

        # Case A: Multiple positions on the exchange. This indicates a serious issue.
        if len(exchange_open_positions) > 1:
            logger.critical(f"Multiple positions found for {self.symbol} on exchange. Manual check recommended.")
            await self.notifier.send_notification("Critical error: Multiple positions found on exchange. Shutting down.", level='critical')
            raise ConfigurationError("Multiple positions detected. Manual check recommended.")

        exchange_pos = exchange_open_positions[0] if exchange_open_positions else None
        
        # Case B: Orphan position on exchange (not in bot state). Adopt it.
        if exchange_pos and not bot_position:
            logger.warning("No bot position on record, but a position found on exchange. Adopting it.")
            
            # Clear all existing orders for the symbol
            if exchange_open_orders:
                orphan_order_ids = [o.get('orderId') for o in exchange_open_orders]
                logger.warning(f"Found and cancelling {len(orphan_order_ids)} existing orders for the adopted position.")
                await self.exchange_adapter.cancel_multiple_orders(self.symbol, orphan_order_ids)
            
            # Create a new bot position dict from the exchange data
            default_max_holding = app_config.trading.sltp.max_holding_period_bars_default or 0
            adopted_position = {
                'symbol': self.symbol,
                'direction_int': 1 if exchange_pos.get('direction') == 'long' else -1,
                'direction_str': exchange_pos.get('direction'),
                'side': 'BUY' if exchange_pos.get('direction') == 'long' else 'SELL',
                'entry_price': exchange_pos.get('entryPrice'),
                'quantity': exchange_pos.get('quantity'),
                'notional_value': exchange_pos.get('entryPrice', 0) * exchange_pos.get('quantity', 0),
                'liquidation_price': exchange_pos.get('liquidationPrice'),
                'initial_margin': exchange_pos.get('entryMargin'),
                'entry_time': exchange_pos.get('entryTime'),
                # Defaults/unknowns:
                'entry_fee': 0.0,
                'holding_period': 0,
                'max_holding_period': default_max_holding,
                'model_probabilities': {},
                'entry_reason': 'adopted_from_exchange',
                # These will be filled in below:
                'stop_loss_price': None,
                'take_profit_price': None,
                'entry_order_id': None,
                'sl_order_id': None,
                'tp_order_id': None,
                'trade_id': str(uuid.uuid4()),
            }

            self.session_manager.set_open_position(adopted_position)
            
            # Place new SL and TP orders for the adopted position.
            await self.trade_cycle_processor.ensure_sltp_orders(position=adopted_position)
            await self.notifier.send_notification(
                f"🚀 TRADE ADOPTED: {adopted_position.get('direction_str').upper()} {self.symbol}\n"
                f"Entry @ {adopted_position.get('entry_price',0):.4f}\n"
                f"Qty: {adopted_position.get('quantity',0.0):.4f}\n"
                f"SL: {adopted_position.get('stop_loss_price',0):.4f} | TP: {adopted_position.get('take_profit_price',0):.4f}",
                level='info'
            )
            logger.info("Reconciliation complete. Orphan position successfully adopted with new SL/TP orders.")

        # Case C: Ghost position in bot state (not on exchange). Clear bot state.
        elif not exchange_pos and bot_position:
            logger.warning(f"Ghost position found in bot state but not on exchange. Clearing bot state and cancelling any open orders.")
            if exchange_open_orders:
                orphan_ids = [o.get('orderId') for o in exchange_open_orders]
                await self.exchange_adapter.cancel_multiple_orders(self.symbol, orphan_ids)
            self.session_manager.clear_open_position()
            logger.info("Reconciliation complete. Bot is now on a clean slate.")
            await self.notifier.send_notification("Bot is starting from a clean slate.")

        # Case D: Both bot and exchange have a position. Sync and ensure SL/TP.
        elif exchange_pos and bot_position:
            logger.info("Bot and exchange positions match. Reconciling details and orders.")
            
            # First, update the bot's state from the exchange
            self.session_manager.load_state_from_dict(bot_state)

            # Cancel any extraneous orders for this symbol.
            legitimate_order_ids = set([str(bot_position.get('sl_order_id')), str(bot_position.get('tp_order_id'))])
            orphan_orders = [o for o in exchange_open_orders if str(o.get('orderId')) not in legitimate_order_ids]
            if orphan_orders:
                orphan_ids = [o.get('orderId') for o in orphan_orders]
                logger.warning(f"Found and cancelling {len(orphan_ids)} orphan orders not tied to the bot's position.")
                await self.exchange_adapter.cancel_multiple_orders(self.symbol, orphan_ids)

            # Now, ensure SL/TP orders exist on the exchange, placing them if they are missing.
            await self.trade_cycle_processor.ensure_sltp_orders(
                position=bot_position
            )
            
            logger.info("Reconciliation complete. Bot is in sync with the exchange and ready to resume.")
            await self.notifier.send_notification("Bot has successfully reconciled its position and is back online.")

        # Case E: Clean slate. Nothing on bot or exchange.
        else:
            logger.info("Bot state and exchange are clean. No reconciliation needed.")
            if exchange_open_orders:
                orphan_ids = [o.get('orderId') for o in exchange_open_orders]
                logger.warning(f"Found and cancelling {len(orphan_ids)} orphan orders for a clean start.")
                await self.exchange_adapter.cancel_multiple_orders(self.symbol, orphan_ids)

            self.session_manager.clear_open_position()
            logger.info("Startup complete. Trading from a clean slate.")
            await self.notifier.send_notification("Bot is starting from a clean slate.")

        return self.session_manager.get_last_processed_timestamp()

    async def shutdown(self):
        """Executes the graceful shutdown sequence."""
        logger.info("--- Bot Lifecycle: Shutdown Phase ---")
        open_position = self.session_manager.get_open_position()
        if open_position:
            logger.info(f"Closing open position {open_position.get( 'trade_id')} due to shutdown...")
            try:
                latest_price = await self.exchange_adapter.get_latest_price(self.symbol)
                await self.trade_cycle_processor.execute_close_workflow(open_position, "graceful_shutdown", latest_price)
            except ExchangeConnectionError as e:
                logger.error(f"Could not get latest price during shutdown. Unable to close position cleanly: {e}")

        self.trade_cycle_processor.save_current_state("shutdown")
        await self.exchange_adapter.close_connection()
        logger.info("Final state saved. Bot shutdown complete.")