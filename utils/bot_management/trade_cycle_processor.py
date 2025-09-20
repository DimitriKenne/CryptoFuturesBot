import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import pandas as pd
import uuid  

from utils.data_management.data_manager import DataManager
from utils.exchange_adapters.exchange_interface import ExchangeInterface
from utils.strategy_execution.live_trading_session_manager import LiveTradingSessionManager
from utils.strategy_execution.trade_execution_engine import TradeExecutionEngine
from utils.data_management.market_data_handler import MarketDataHandler
from utils.exceptions import ExchangeConnectionError, OrderExecutionError
from utils.notification_manager import NotificationManager

logger = logging.getLogger(__name__)

class TradeCycleProcessor:
    """Handles the logic for a single trading cycle within the main bot loop."""

    def __init__(
        self,
        data_manager: DataManager,
        market_data_handler: MarketDataHandler,
        exchange_adapter: ExchangeInterface,
        session_manager: LiveTradingSessionManager,
        trade_execution_engine: TradeExecutionEngine,
        notifier: NotificationManager,
        model_type: str,
        symbol: str,
        interval: str,
        mode: str
    ):
        self.data_manager = data_manager
        self.market_data_handler = market_data_handler
        self.exchange_adapter = exchange_adapter
        self.session_manager = session_manager
        self.trade_execution_engine = trade_execution_engine
        self.notifier = notifier
        self.model_type = model_type
        self.symbol = symbol
        self.interval = interval
        self.mode = mode  # 'automatic' or 'hybrid'
        self.last_processed_timestamp: Optional[datetime] = None

    def set_last_processed_timestamp(self, timestamp: Optional[datetime]):
        self.last_processed_timestamp = timestamp

    async def process_cycle(self):
        """Executes one full trading cycle."""
        logger.debug("--- Main Loop Cycle Start ---")
        
        latest_candle_data = await self.market_data_handler.get_latest_data(
            self.exchange_adapter, self.last_processed_timestamp
        )
        if latest_candle_data is None:
            return

        open_position = self.session_manager.get_open_position()
        if open_position:
            await self._handle_exits(open_position, latest_candle_data)
        else:
            await self._handle_new_entry(latest_candle_data)
        
        self.last_processed_timestamp = latest_candle_data.name.to_pydatetime()
        self.save_current_state("cycle_end")

    async def _handle_exits(self, position: Dict[str, Any], candle_data: pd.Series):
        """Checks for an exit condition and executes the close if triggered."""
        exit_triggered, reason, exit_price = self.trade_execution_engine.check_exit_conditions(position, candle_data, -1)
        if exit_triggered:
            logger.info(f"Exit triggered for position {position.get('trade_id', '<no-id>')} due to: {reason}")
            await self.execute_close_workflow(position, reason, exit_price)

    async def execute_close_workflow(self, position: Dict[str, Any], reason: str, exit_price: float):
        """
        Implements the full close position workflow with exchange confirmation and orphan order cleanup.
        Now updated: Only clears the position and sends 'closed' notification if position is confirmed closed on exchange.
        """
        logger.info(f"Executing close workflow for position {position.get('trade_id', '<no-id>')}.")

        try:
            # 1. Attempt to cancel remaining SL/TP order(s)
            try:
                await self.trade_execution_engine.cancel_remaining_orders(position, reason)
            except Exception as e:
                logger.error(f"Error canceling SL/TP orders for position {position.get('trade_id', '<no-id>')}: {e}")

            # 2. Check if position is still open on exchange
            try:
                open_positions = await self.exchange_adapter.get_open_positions(self.symbol)
            except Exception as e:
                logger.error(f"Error fetching open positions: {e}")
                await self.notifier.send_notification(
                    f"🚨 ERROR: Unable to fetch open positions for {self.symbol}: {e}", level="critical"
                )
                # Do NOT mark as closed, return
                return

            position_still_open = False
            for pos in open_positions:
                if pos.get('symbol', '').upper() == self.symbol.upper() and float(pos.get('quantity', 0)) > 0:
                    position_still_open = True
                    break

            if position_still_open:
                close_side = 'SELL' if position['direction_str'] == 'long' else 'BUY'
                try:
                    close_order = await self.exchange_adapter.place_market_order(
                        symbol=self.symbol, side=close_side,
                        quantity=position['quantity'], reduce_only=True
                    )
                    order_id = close_order.get('orderId')
                    elapsed = 0
                    max_wait = 15  # seconds
                    interval = 1.0 # seconds
                    order_info = close_order
                    while order_info.get('status') != 'FILLED' and elapsed < max_wait:
                        await asyncio.sleep(interval)
                        elapsed += interval
                        order_info = await self.exchange_adapter.get_order_info(self.symbol, order_id)

                    if order_info.get('status') == 'FILLED':
                        # Mark as closed and send notification
                        finalized_trade = self.trade_execution_engine.calculate_exit_details(
                            open_trade=position, exit_price=order_info['avgPrice'],
                            exit_time=order_info['time'], exit_reason=reason
                        )
                        self.session_manager.close_position(finalized_trade)
                        await self.notifier.send_notification(
                            f"✅ TRADE CLOSED: {position.get('direction_str').upper()} {self.symbol}\n"
                            f"Exit @ {finalized_trade.get('exit_price',0):.4f}\n"
                            f"Net PnL: ${finalized_trade.get('net_pnl',0.0):.2f}\n"
                            f"Reason: {reason}\n"
                            f"Capital: ${self.session_manager.get_current_capital():.2f}",
                            level='info'
                        )
                    else:
                        logger.error(f"Market close order {order_id} not filled after {max_wait} seconds. Position NOT marked as closed.")
                        await self.notifier.send_notification(
                            f"🚨 ERROR: Market close order {order_id} not filled after {max_wait}s. Trade NOT finalized. Position remains open.",
                            level='critical'
                        )
                        return
                except Exception as e:
                    logger.error(f"Error executing market close for position {position.get('trade_id', '<no-id>')}: {e}", exc_info=True)
                    await self.notifier.send_notification(
                        f"🚨 ERROR closing trade {position.get('trade_id', '<no-id>')}: {e} Position remains open.", level='critical'
                    )
                    return

            else:
                # Already closed on exchange, get actual fill info if possible
                logger.info(f"No open position found for {self.symbol} on exchange. Attempting to fetch order fill info for accurate exit details.")

                order_id = None
                exit_fill_info = None
                if reason == 'stop_loss' and position.get("sl_order_id"):
                    order_id = position["sl_order_id"]
                elif reason == 'take_profit' and position.get("tp_order_id"):
                    order_id = position["tp_order_id"]

                if order_id:
                    try:
                        exit_fill_info = await self.exchange_adapter.get_order_info(self.symbol, order_id)
                    except Exception as e:
                        logger.warning(f"Could not fetch fill info for {reason} order {order_id}: {e}")

                if exit_fill_info and exit_fill_info.get("status") == "FILLED":
                    finalized_trade = self.trade_execution_engine.calculate_exit_details(
                        open_trade=position,
                        exit_price=exit_fill_info['avgPrice'],
                        exit_time=exit_fill_info['time'],
                        exit_reason=reason
                    )
                    self.session_manager.close_position(finalized_trade)
                    await self.notifier.send_notification(
                        f"✅ TRADE CLOSED: {position.get('direction_str').upper()} {self.symbol}\n"
                        f"Exit @ {finalized_trade.get('exit_price',0):.4f}\n"
                        f"Net PnL: ${finalized_trade.get('net_pnl',0.0):.2f}\n"
                        f"Reason: {reason}\n"
                        f"Capital: ${self.session_manager.get_current_capital():.2f}",
                        level='info'
                    )
                else:
                    # Fallback: Do NOT mark as closed, send error notification
                    logger.warning(f"Could not confirm fill info for exit. Position NOT marked as closed.")
                    await self.notifier.send_notification(
                        f"🚨 ERROR: Could not confirm fill info for exit. Position NOT marked as closed for {self.symbol}.",
                        level='critical'
                    )
                    return

            # 5. Cleanup orphan orders after closure
            try:
                open_orders = await self.exchange_adapter.get_open_orders(self.symbol)
                orphan_order_ids = [o['orderId'] for o in open_orders]
                if orphan_order_ids:
                    await self.exchange_adapter.cancel_multiple_orders(self.symbol, orphan_order_ids)
                    logger.info(f"Cancelled orphan orders for {self.symbol}: {orphan_order_ids}")
                    await self.notifier.send_notification(
                        f"Cancelled orphan orders for {self.symbol}: {orphan_order_ids}", level='info'
                    )
            except Exception as e:
                logger.error(f"Error cleaning up orphan orders: {e}")

        except Exception as e:
            logger.error(f"Error executing close workflow for position {position.get( 'trade_id', '<no-id>')}: {e}", exc_info=True)
            await self.notifier.send_notification(
                f"🚨 ERROR during close workflow for trade {position.get( 'trade_id', '<no-id>')}: {e}. Position NOT marked as closed.",
                level='critical'
            )
         
    async def _handle_new_entry(self, candle_data: pd.Series):
        """Checks for a new entry signal and executes the entry workflow."""
        signal = int(candle_data.get('signal', 0))
        if signal == 0:
            return
        if not self.session_manager.can_open_new_trade(candle_data.name):
            return
        probs = candle_data.get('probabilities', {})
        model_probabilities = {k: float(probs.get(k, 0.0)) for k in [-1, 0, 1]}
        trade_plan = self.trade_execution_engine.calculate_entry_details(
            signal=signal,
            current_capital=self.session_manager.get_current_capital(),
            current_price=candle_data['close'],
            current_bar_features=candle_data,
            model_probabilities=pd.Series(model_probabilities)
        )
        
        if not trade_plan:
            return
        
        # --- NEW: Generate a unique trade_id for this trade session ---
        trade_id = str(uuid.uuid4())
        trade_plan['trade_id'] = trade_id
        logger.info(f"New trade initiated with unique ID: {trade_id}")
        
        if self.mode == "hybrid":
            confirmed = await self.hybrid_menu(trade_plan)
            if not confirmed:
                logger.info("Trade entry rejected/skipped by user in hybrid mode.")
                await self.notifier.send_notification(
                    "Trade entry rejected/skipped by user.", level="info"
                )
                return
        
        await self._execute_entry_workflow(trade_plan)

    async def _execute_entry_workflow(self, trade_plan: Dict[str, Any]):
        """Implements the full new entry workflow with verification."""
        logger.info(f"Executing entry workflow for a {trade_plan['direction_str']} trade.")
        entry_order, sl_order, tp_order = None, None, None
        try:
            entry_order = await self.exchange_adapter.place_market_order(
                self.symbol, trade_plan['side'], trade_plan['quantity']
            )
            order_id = entry_order['orderId']
            max_wait = 30  # seconds
            interval = 1.5   # seconds
            elapsed = 0
            order_info = entry_order
            while order_info.get('status') != 'FILLED' and elapsed < max_wait:
                await asyncio.sleep(interval)
                elapsed += interval
                order_info = await self.exchange_adapter.get_order_info(self.symbol, order_id)
            if order_info.get('status') != 'FILLED':
                logger.error(f"Order {order_id} not filled after {max_wait} seconds. Aborting entry workflow.")
                await self.notifier.send_notification(f"🚨 ERROR: Order {order_id} not filled after {max_wait}s.", level='error')
                await self.trade_execution_engine.cleanup_failed_entry(self.symbol, entry_order, None, None)
                return

            sl_order, tp_order = await self.trade_execution_engine.place_and_verify_sltp_orders(trade_plan)
            liq_price = await self.exchange_adapter.get_position_liquidation_price(self.symbol)
            final_position = self.trade_execution_engine.reconcile_open_position(
                trade_plan, order_info, sl_order, tp_order, liq_price
            )
            self.session_manager.set_open_position(final_position)
            await self.notifier.send_notification(
                f"🚀 TRADE ENTERED: {final_position.get('direction_str').upper()} {self.symbol}\n"
                f"Entry @ {final_position.get('entry_price',0):.4f}\n"
                f"Qty: {final_position.get('quantity',0.0):.4f}\n"
                f"SL: {final_position.get('stop_loss_price',0):.4f} | TP: {final_position.get('take_profit_price',0):.4f}",
                level='info'
            )
        except (OrderExecutionError, ExchangeConnectionError) as e:
            logger.critical(f"CRITICAL FAILURE in entry workflow: {e}", exc_info=True)
            await self.notifier.send_notification(f"🚨 CRITICAL ERROR entering trade: {e}", level='critical')
            await self.trade_execution_engine.cleanup_failed_entry(self.symbol, entry_order, sl_order, tp_order)

    def save_current_state(self, reason: str):
        """Saves the bot's current state to a file."""
        self.session_manager.set_last_processed_timestamp(self.last_processed_timestamp)
        self.data_manager.save_bot_state(
            self.session_manager.get_state_as_dict(),
            self.model_type, self.symbol, self.interval
        )

    async def hybrid_menu(self, trade_plan: Dict[str, Any], timeout: int = 60):
        """CLI menu for hybrid mode with timeout and input validation."""
        import threading

        menu_text = (
            "\n--- Trade Proposal ---\n"
            f"Direction: {trade_plan.get('direction_str','').upper()}\n"
            f"Price: {trade_plan.get('entry_price')}\n"
            f"Quantity: {trade_plan.get('quantity')}\n"
            f"SL: {trade_plan.get('stop_loss_price')}\n"
            f"TP: {trade_plan.get('take_profit_price')}\n"
            "----------------------\n"
            "Options:\n"
            "  y      - Approve and enter trade\n"
            "  n      - Reject trade\n"
            "  help   - Print trade details again\n"
            "  skip   - Skip and go to next cycle\n"
            f"(Timeout in {timeout} seconds will skip)\n"
        )

        # --- Notify via Telegram that trade needs approval ---
        notification_text = (
            "🚦 Trade Approval Needed (Hybrid Mode):\n"
            f"Symbol: {self.symbol}\n"
            f"Direction: {trade_plan.get('direction_str','').upper()}\n"
            f"Price: {trade_plan.get('entry_price')}\n"
            f"Quantity: {trade_plan.get('quantity')}\n"
            f"SL: {trade_plan.get('stop_loss_price')}\n"
            f"TP: {trade_plan.get('take_profit_price')}\n"
            "Reply in terminal to approve or reject."
        )
        logger.info("Trade proposal requires user approval (hybrid mode).")
        await self.notifier.send_notification(notification_text, level="info")

        valid_responses = {'y', 'n', 'help', 'skip'}
        response = None

        def prompt():
            nonlocal response
            print(menu_text)
            while True:
                raw = input("Your choice: ").strip().lower()
                if raw in valid_responses:
                    response = raw
                    break
                else:
                    print("Invalid input. Type 'help' for menu.")

        # Run prompt in thread for timeout
        import threading
        prompt_thread = threading.Thread(target=prompt)
        prompt_thread.daemon = True
        prompt_thread.start()
        prompt_thread.join(timeout)
        if response is None:
            logger.warning(f"No input received in {timeout} seconds. Skipping trade.")
            await self.notifier.send_notification(
                f"Trade entry skipped due to timeout ({timeout} seconds).", level="warning"
            )
            return False
        if response == 'y':
            logger.info("Trade entry approved by user in hybrid mode.")
            await self.notifier.send_notification(
                "Trade entry approved by user.", level="info"
            )
            return True
        elif response == 'n' or response == 'skip':
            logger.info("Trade entry rejected/skipped by user in hybrid mode.")
            await self.notifier.send_notification(
                "Trade entry rejected/skipped by user.", level="info"
            )
            return False
        elif response == 'help':
            print(menu_text)
            return await self.hybrid_menu(trade_plan, timeout)
        return False

    # Ensure SL/TP orders are in place for a live position during the reconciliation startup
    async def ensure_sltp_orders(self, position: Dict[str, Any]):
        """Detects and replaces missing SL/TP orders for a live position."""

        sl_missing = not position.get("sl_order_id")
        tp_missing = not position.get("tp_order_id")
        
        if not sl_missing and not tp_missing:
            logger.info("Both SL and TP orders already exist. No action needed.")
            return
        
        symbol = position["symbol"]
        side = position["side"]  # Use side directly
        side_to_close = 'SELL' if side == 'BUY' else 'BUY'
        quantity = position["quantity"]

        entry_price = position.get("entry_price") or position.get("entryPrice")
        latest_atr = None # Optionally fetch ATR from latest candle or features if needed
        stop_loss_price, take_profit_price = self.trade_execution_engine.trade_calculation_helpers.calculate_sl_tp_prices(
            side=side.lower(), current_price=entry_price, latest_atr=latest_atr
        )
        position["stop_loss_price"] = stop_loss_price
        position["take_profit_price"] = take_profit_price
        
        # Place missing SL
        if sl_missing:
            logger.info(f"SL order ID not found. Placing new SL order for position {position.get('trade_id', '<no-id>')} at price {stop_loss_price}.")
            sl_order = await self.exchange_adapter.place_stop_market_order(
                symbol=symbol, side=side_to_close, quantity=quantity, stop_price=stop_loss_price
            )
            position["sl_order_id"] = sl_order["orderId"]

        # Place missing TP
        if tp_missing:
            logger.info(f"TP order ID not found. Placing new TP order for position {position.get('trade_id', '<no-id>')} at price {take_profit_price}.")
            tp_order = await self.exchange_adapter.place_take_profit_market_order(
                symbol=symbol, side=side_to_close, quantity=quantity, stop_price=take_profit_price
            )
            position["tp_order_id"] = tp_order["orderId"]

        # Update the session manager state
        self.session_manager.set_open_position(position)

        self.save_current_state("sl_tp_reconciliation_complete")
        