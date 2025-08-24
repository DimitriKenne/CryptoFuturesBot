# utils/bot_management/lifecycle_manager.py

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
        """Implements the robust two-phase reconciliation logic."""
        logger.info("--- Initiating State Reconciliation ---")
        
        # --- Phase A: Fetch Ground Truth ---
        exchange_positions = await self.exchange_adapter.get_open_positions(self.symbol)
        exchange_orders = await self.exchange_adapter.get_open_orders(self.symbol)
        bot_position = self.session_manager.get_open_position()

        # --- Phase B: Reconcile State & Enforce Protection ---
        
        # 1. Reconcile Active Position
        if len(exchange_positions) > 1:
            logger.critical(f"Multiple positions found for {self.symbol} on exchange. Manual intervention required.")
            raise ConfigurationError("Multiple positions detected. Cannot reconcile automatically.")

        exchange_pos = exchange_positions[0] if exchange_positions else None

        if exchange_pos and not bot_position:
            logger.critical(f"Orphan position found on exchange but not in bot state. Manual intervention required.")
            raise ConfigurationError(f"Orphan position detected: {exchange_pos}")

        if not exchange_pos and bot_position:
            logger.warning(f"Ghost position found in bot state but not on exchange. Clearing bot state.")
            self.session_manager.clear_open_position()
            bot_position = None

        if exchange_pos and bot_position:
            logger.info("Bot and exchange positions match. Adopting exchange state.")
            # Update bot position with the latest from exchange (e.g., unrealized PnL)
            bot_position['entryPrice'] = exchange_pos['entryPrice']
            bot_position['quantity'] = exchange_pos['quantity']
            self.session_manager.set_open_position(bot_position)

        # 2. Reconcile SL/TP Orders
        if bot_position:
            ideal_sl_id = bot_position.get('sl_order_id')
            ideal_tp_id = bot_position.get('tp_order_id')
            
            # Check if ideal orders exist
            sl_exists = any(o['orderId'] == ideal_sl_id for o in exchange_orders)
            tp_exists = any(o['orderId'] == ideal_tp_id for o in exchange_orders)

            if not sl_exists:
                logger.warning(f"Missing SL order for position {bot_position['id']}. Re-placing now.")
                # Re-place logic would go here, calling trade_execution_engine
                # For now, we raise an error to be safe
                raise ConfigurationError("Missing SL order on startup. Needs manual check.")

            if not tp_exists:
                logger.warning(f"Missing TP order for position {bot_position['id']}. Re-placing now.")
                raise ConfigurationError("Missing TP order on startup. Needs manual check.")
                
            # 3. Cleanup Orphan Orders
            legitimate_order_ids = {ideal_sl_id, ideal_tp_id}
            orphan_orders = [o for o in exchange_orders if o['orderId'] not in legitimate_order_ids]
            if orphan_orders:
                orphan_ids = [o['orderId'] for o in orphan_orders]
                logger.warning(f"Found {len(orphan_ids)} orphan orders. Cancelling them: {orphan_ids}")
                await self.exchange_adapter.cancel_multiple_orders(self.symbol, orphan_ids)

        elif not bot_position and exchange_orders:
             # No position, so all orders for this symbol are orphans
             orphan_ids = [o['orderId'] for o in exchange_orders]
             logger.warning(f"No active position, but found {len(orphan_ids)} open orders. Cancelling all.")
             await self.exchange_adapter.cancel_all_orders(self.symbol)

        logger.info("--- State Reconciliation Complete ---")

    async def shutdown(self, trade_cycle_processor: 'TradeCycleProcessor'):
        """Executes the graceful shutdown sequence."""
        logger.info("--- Bot Lifecycle: Shutdown Phase ---")
        open_position = self.session_manager.get_open_position()
        if open_position:
            logger.info(f"Closing open position {open_position['id']} due to shutdown...")
            try:
                latest_price = await self.exchange_adapter.get_latest_price(self.symbol)
                await trade_cycle_processor.execute_close_workflow(open_position, "graceful_shutdown", latest_price)
            except ExchangeConnectionError as e:
                logger.error(f"Could not get latest price during shutdown. Unable to close position cleanly: {e}")

        trade_cycle_processor.save_current_state("shutdown")
        await self.exchange_adapter.close_connection()
        logger.info("Final state saved. Bot shutdown complete.")