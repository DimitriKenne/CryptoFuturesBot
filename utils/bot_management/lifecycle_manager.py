import logging
from typing import TYPE_CHECKING, Optional

from config.params import app_config
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
        trade_cycle_processor: 'TradeCycleProcessor'
    ):
        self.data_manager = data_manager
        self.exchange_adapter = exchange_adapter
        self.session_manager = session_manager
        self.model_type = model_type
        self.symbol = symbol
        self.interval = interval
        self.trade_cycle_processor = trade_cycle_processor

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
        - Uses trade_cycle_processor.ensure_sltp_orders() to recreate missing SL/TP.
        - Closes orphan/ghost positions using exchange_adapter.close_position.
        - Cleans up orphan orders.
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

        # 2. Orphan position on exchange (not in bot state): ADOPT and ensure SL/TP
        exchange_pos = exchange_positions[0] if exchange_positions else None
        default_max_holding = app_config.trading.sltp.max_holding_period_bars_default or 0
        if exchange_pos and not bot_position:
            logger.warning("Orphan position found on exchange but not in bot state. Attempting adoption.")
            logger.info(f"Exchange Position: {exchange_pos}")
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
                'max_holding_bars': default_max_holding,
                'model_probabilities': {},
                'entry_reason': 'adopted_from_exchange',
                # These will be filled in below:
                'stop_loss_price': None,
                'take_profit_price': None,
                'entry_order_id': None,
                'sl_order_id': None,
                'tp_order_id': None,
                'id': None,
            }

            # Now, loop for SL/TP orders:
            sl_order, tp_order = None, None
            for order in exchange_orders:
                if order.get('type') in ('STOP_MARKET', 'STOP'):
                    sl_order = order
                if order.get('type') in ('TAKE_PROFIT_MARKET', 'TAKE_PROFIT'):
                    tp_order = order

            if sl_order:
                adopted_position['sl_order_id'] = sl_order['orderId']
                adopted_position['stop_loss_price'] = sl_order.get('stopPrice')
            if tp_order:
                adopted_position['tp_order_id'] = tp_order['orderId']
                adopted_position['take_profit_price'] = tp_order.get('stopPrice')

            adopted_position['id'] = adopted_position['entry_order_id'] or adopted_position['sl_order_id'] or adopted_position['tp_order_id'] or f"adopted_{self.symbol}"

            self.session_manager.set_open_position(adopted_position)
            self.data_manager.save_bot_state(self.session_manager.get_state_as_dict(), self.model_type, self.symbol, self.interval)
            logger.info("Successfully adopted exchange position into bot state.")

            # If SL/TP missing, use trade_cycle_processor.ensure_sltp_orders()
            if not sl_order or not tp_order:
                logger.warning("Missing SL or TP order after adoption. Attempting to replace via ensure_sltp_orders().")
                await self.trade_cycle_processor.ensure_sltp_orders(adopted_position)
                logger.info("Successfully recreated missing SL/TP orders after adoption.")

            return

        # 3. Ghost position in bot state (not on exchange): clear bot state
        if not exchange_pos and bot_position:
            logger.warning(f"Ghost position found in bot state but not on exchange. Clearing bot state.")
            self.session_manager.clear_open_position()
            bot_position = None

        # 4. Both bot and exchange have a position: sync details and ensure SL/TP
        if exchange_pos and bot_position:
            logger.info("Bot and exchange positions match. Adopting exchange state.")
            bot_position['entryPrice'] = exchange_pos.get('entryPrice')
            bot_position['quantity'] = exchange_pos.get('quantity')
            self.session_manager.set_open_position(bot_position)
            self.data_manager.save_bot_state(self.session_manager.get_state_as_dict(), self.model_type, self.symbol, self.interval)

        # 5. Reconcile SL/TP Orders: recreate if missing
        if bot_position:
            ideal_sl_id = bot_position.get('sl_order_id')
            ideal_tp_id = bot_position.get('tp_order_id')

            sl_exists = any(str(o.get('orderId')) == str(ideal_sl_id) for o in exchange_orders)
            tp_exists = any(str(o.get('orderId')) == str(ideal_tp_id) for o in exchange_orders)

            if not sl_exists or not tp_exists:
                logger.warning(f"Missing SL or TP order for position {bot_position.get('id', 'unknown')}. Attempting to recreate via ensure_sltp_orders().")
                await self.trade_cycle_processor.ensure_sltp_orders(bot_position)
                logger.info("Successfully recreated missing SL/TP orders for bot position.")

            # Cleanup orphan orders (orders not in bot state)
            legitimate_order_ids = set([str(bot_position.get('sl_order_id')), str(bot_position.get('tp_order_id'))])
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

    async def shutdown(self):
        """Executes the graceful shutdown sequence."""
        logger.info("--- Bot Lifecycle: Shutdown Phase ---")
        open_position = self.session_manager.get_open_position()
        if open_position:
            logger.info(f"Closing open position {open_position.get('id')} due to shutdown...")
            try:
                latest_price = await self.exchange_adapter.get_latest_price(self.symbol)
                await self.trade_cycle_processor.execute_close_workflow(open_position, "graceful_shutdown", latest_price)
            except ExchangeConnectionError as e:
                logger.error(f"Could not get latest price during shutdown. Unable to close position cleanly: {e}")

        self.trade_cycle_processor.save_current_state("shutdown")
        await self.exchange_adapter.close_connection()
        logger.info("Final state saved. Bot shutdown complete.")