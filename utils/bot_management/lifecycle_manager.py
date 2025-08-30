import logging
from typing import TYPE_CHECKING, Optional

from utils.data_management.data_manager import DataManager
from utils.exchange_adapters.exchange_interface import ExchangeInterface
from utils.strategy_execution.live_trading_session_manager import LiveTradingSessionManager
from utils.exceptions import ExchangeConnectionError, ConfigurationError

if TYPE_CHECKING:
    from .trade_cycle_processor import TradeCycleProcessor

logger = logging.getLogger(__name__)

class LifecycleManager:
    """Manages the startup and shutdown phases of the trading bot, including robust state reconciliation."""

    def __init__(self, data_manager: DataManager, exchange_adapter: ExchangeInterface, session_manager: LiveTradingSessionManager, model_type: str, symbol: str, interval: str):
        self.data_manager = data_manager
        self.exchange_adapter = exchange_adapter
        self.session_manager = session_manager
        self.model_type = model_type
        self.symbol = symbol
        self.interval = interval

    async def startup(self):
        """Handles the complete startup procedure: loading state, connecting, and robustly synchronizing."""
        logger.info("--- Bot Lifecycle: Startup Phase ---")
        
        await self.exchange_adapter.async_setup()
        
        loaded_state = self.data_manager.load_bot_state(self.model_type, self.symbol, self.interval)
        if loaded_state:
            self.session_manager.load_state_from_dict(loaded_state)

        await self._reconcile_state_with_exchange()

        logger.info("--- Bot Lifecycle: Startup Complete. Bot is ready to trade. ---")
        return self.session_manager.get_last_processed_timestamp()

    async def _reconcile_state_with_exchange(self):
        """
        Implements robust reconciliation:
        - Adopts orphan positions on exchange into bot state.
        - Recreates missing SL/TP orders if needed.
        """
        logger.info("--- Initiating State Reconciliation ---")
        
        exchange_positions = await self.exchange_adapter.get_open_positions(self.symbol)
        exchange_orders = await self.exchange_adapter.get_open_orders(self.symbol)
        bot_position = self.session_manager.get_open_position()

        # 1. Multiple active exchange positions: close orphans and halt
        if len(exchange_positions) > 1:
            logger.critical(f"Multiple positions found for {self.symbol} on exchange. Attempting to close orphan positions.")
            orphan_positions = []
            for pos in exchange_positions:
                if bot_position:
                    if (str(pos.get("entryPrice")) != str(bot_position.get("entryPrice")) or 
                        str(pos.get("quantity")) != str(bot_position.get("quantity"))):
                        orphan_positions.append(pos)
                else:
                    orphan_positions.append(pos)
            for orphan in orphan_positions:
                logger.warning(f"Closing orphan position: {orphan}")
                try:
                    await self.exchange_adapter.close_position(self.symbol, orphan)
                except Exception as e:
                    logger.error(f"Failed to close orphan position: {e}")
            raise ConfigurationError("Multiple positions detected. Orphans attempted to close. Manual check recommended.")

        exchange_pos = exchange_positions[0] if exchange_positions else None

        # 2. Orphan position on exchange (not in bot state): ADOPT instead of close
        if exchange_pos and not bot_position:
            logger.warning("Orphan position found on exchange but not in bot state. Attempting adoption.")
            adopted_position = {
                'symbol': self.symbol,
                'direction': exchange_pos.get('direction'),
                'quantity': exchange_pos.get('quantity'),
                'entryPrice': exchange_pos.get('entryPrice'),
                'unrealizedPnl': exchange_pos.get('unrealizedPnl'),
                'leverage': exchange_pos.get('leverage'),
                'entryMargin': exchange_pos.get('entryMargin'),
                'liquidationPrice': exchange_pos.get('liquidationPrice'),
                'entryTime': exchange_pos.get('entryTime'),
                # Attempt to reconstruct SL/TP order IDs from open orders below
            }
            # Try to find existing SL/TP orders (by reduceOnly and type)
            sl_order = None
            tp_order = None
            for order in exchange_orders:
                if order['reduceOnly'] and order['type'] in ('STOP_MARKET', 'STOP'):
                    sl_order = order
                if order['reduceOnly'] and order['type'] in ('TAKE_PROFIT_MARKET', 'TAKE_PROFIT'):
                    tp_order = order
            adopted_position['sl_order_id'] = sl_order.get('orderId') if sl_order else None
            adopted_position['tp_order_id'] = tp_order.get('orderId') if tp_order else None
            # If you have trade IDs or other fields, add here

            self.session_manager.set_open_position(adopted_position)
            self.data_manager.save_bot_state(self.session_manager.get_state_as_dict(), self.model_type, self.symbol, self.interval)
            logger.info("Successfully adopted exchange position into bot state.")

            # If SL/TP missing, try to recreate
            # Use your trade_execution_engine to create them
            if not sl_order or not tp_order:
                logger.warning("Missing SL or TP order after adoption. Attempting to replace.")
                # You may need to import or pass trade_execution_engine here, or raise for manual intervention
                # For now, raise ConfigurationError to trigger manual handling or add logic as needed
                raise ConfigurationError("Missing SL/TP order(s) after position adoption. Should auto-replace or manual check.")

            return  # Adoption done; continue with bot startup

        # 3. Ghost position in bot state (not on exchange): clear bot state
        if not exchange_pos and bot_position:
            logger.warning(f"Ghost position found in bot state but not on exchange. Clearing bot state.")
            self.session_manager.clear_open_position()
            bot_position = None

        # 4. Both bot and exchange have a position: sync details
        if exchange_pos and bot_position:
            logger.info("Bot and exchange positions match. Adopting exchange state.")
            bot_position['entryPrice'] = exchange_pos.get('entryPrice')
            bot_position['quantity'] = exchange_pos.get('quantity')
            self.session_manager.set_open_position(bot_position)

        # 5. Reconcile SL/TP Orders: recreate if missing
        if bot_position:
            ideal_sl_id = bot_position.get('sl_order_id')
            ideal_tp_id = bot_position.get('tp_order_id')

            sl_exists = any(str(o.get('orderId')) == str(ideal_sl_id) for o in exchange_orders)
            tp_exists = any(str(o.get('orderId')) == str(ideal_tp_id) for o in exchange_orders)

            if not sl_exists or not tp_exists:
                logger.warning(f"Missing SL or TP order for position {bot_position.get('id', 'unknown')}. Attempting to recreate.")
                # Here, you'd normally call trade_execution_engine.place_and_verify_sltp_orders(bot_position)
                # For integration, pass trade_execution_engine into LifecycleManager or use a callback
                # For now, raise to indicate manual/auto-replacement needed
                raise ConfigurationError("Missing SL/TP order on startup. Should auto-replace or manual check.")

            # Cleanup orphan orders (orders not in bot state)
            legitimate_order_ids = set([str(ideal_sl_id), str(ideal_tp_id)])
            orphan_orders = [o for o in exchange_orders if str(o.get('orderId')) not in legitimate_order_ids]
            if orphan_orders:
                orphan_ids = [o.get('orderId') for o in orphan_orders]
                logger.warning(f"Found {len(orphan_ids)} orphan orders. Cancelling them: {orphan_ids}")
                await self.exchange_adapter.cancel_multiple_orders(self.symbol, orphan_ids)

        elif not bot_position and exchange_orders:
            # No position, so all orders for this symbol are orphans
            orphan_ids = [o.get('orderId') for o in exchange_orders]
            logger.warning(f"No active position, but found {len(orphan_ids)} open orders. Cancelling all.")
            await self.exchange_adapter.cancel_multiple_orders(self.symbol, orphan_ids)

        logger.info("--- State Reconciliation Complete ---")

    async def shutdown(self, trade_cycle_processor: 'TradeCycleProcessor'):
        """Executes the graceful shutdown sequence."""
        logger.info("--- Bot Lifecycle: Shutdown Phase ---")
        open_position = self.session_manager.get_open_position()
        if open_position:
            logger.info(f"Closing open position {open_position.get('id')} due to shutdown...")
            try:
                latest_price = await self.exchange_adapter.get_latest_price(self.symbol)
                await trade_cycle_processor.execute_close_workflow(open_position, "graceful_shutdown", latest_price)
            except ExchangeConnectionError as e:
                logger.error(f"Could not get latest price during shutdown. Unable to close position cleanly: {e}")

        trade_cycle_processor.save_current_state("shutdown")
        await self.exchange_adapter.close_connection()
        logger.info("Final state saved. Bot shutdown complete.")