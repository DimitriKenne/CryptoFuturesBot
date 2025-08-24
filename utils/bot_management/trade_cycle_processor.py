# utils/bot_management/trade_cycle_processor.py

import logging
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

from utils.data_management.data_manager import DataManager
from utils.exchange_adapters.exchange_interface import ExchangeInterface
from utils.strategy_execution.live_trading_session_manager import LiveTradingSessionManager
from utils.strategy_execution.trade_execution_engine import TradeExecutionEngine
from utils.data_management.market_data_handler import MarketDataHandler
from utils.exceptions import ExchangeConnectionError, OrderExecutionError
from utils.notification_manager import NotificationManager  # <-- ADDED IMPORT

logger = logging.getLogger(__name__)

class TradeCycleProcessor:
    """Handles the logic for a single trading cycle within the main bot loop."""

    def __init__(self, data_manager: DataManager, market_data_handler: MarketDataHandler, 
                 exchange_adapter: ExchangeInterface, session_manager: LiveTradingSessionManager, 
                 trade_execution_engine: TradeExecutionEngine, notifier: NotificationManager, # <-- ADDED notifier
                 model_type: str, symbol: str, interval: str):
        self.data_manager = data_manager
        self.market_data_handler = market_data_handler
        self.exchange_adapter = exchange_adapter
        self.session_manager = session_manager
        self.trade_execution_engine = trade_execution_engine
        self.notifier = notifier  # <-- ADDED notifier
        self.model_type = model_type
        self.symbol = symbol
        self.interval = interval
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
        # Note: In live mode, current_bar_index is not used, so we pass -1.
        exit_triggered, reason, exit_price = self.trade_execution_engine.check_exit_conditions(position, candle_data, -1)
        if exit_triggered:
            logger.info(f"Exit triggered for position {position['id']} due to: {reason}")
            await self.execute_close_workflow(position, reason, exit_price)

    async def execute_close_workflow(self, position: Dict[str, Any], reason: str, exit_price: float):
        """Implements the full close position workflow."""
        logger.info(f"Executing close workflow for position {position['id']}.")
        try:
            await self.trade_execution_engine.cancel_remaining_orders(position, reason)
            
            close_side = 'SELL' if position['direction_str'] == 'long' else 'BUY'
            close_confirmation = await self.exchange_adapter.place_market_order(
                symbol=self.symbol, side=close_side,
                quantity=position['quantity'], reduce_only=True
            )
            
            finalized_trade = self.trade_execution_engine.calculate_exit_details(
                open_trade=position, exit_price=close_confirmation['avgPrice'],
                exit_time=close_confirmation['time'], exit_reason=reason
            )
            
            self.session_manager.close_position(finalized_trade)
            
            # --- ADDED NOTIFICATION ---
            await self.notifier.send_notification(
                f"✅ TRADE CLOSED: {position.get('direction_str').upper()} {self.symbol}\n"
                f"Exit @ {finalized_trade.get('exit_price',0):.4f}\n"
                f"Net PnL: ${finalized_trade.get('net_pnl',0.0):.2f}\n"
                f"Reason: {reason}\n"
                f"Capital: ${self.session_manager.get_current_capital():.2f}",
                level='info'
            )
            
        except (OrderExecutionError, ExchangeConnectionError) as e:
            logger.error(f"Error executing close workflow for position {position['id']}: {e}", exc_info=True)
            await self.notifier.send_notification(f"🚨 ERROR closing trade {position['id']}: {e}", level='error')

    async def _handle_new_entry(self, candle_data: pd.Series):
        """Checks for a new entry signal and executes the entry workflow."""
        signal = int(candle_data.get('signal', 0))
        if signal == 0: return

        if not self.session_manager.can_open_new_trade(candle_data.name):
            return

        trade_plan = self.trade_execution_engine.calculate_entry_details(
            signal=signal, current_capital=self.session_manager.get_current_capital(),
            current_price=candle_data['close'], current_bar_features=candle_data,
            model_probabilities=pd.Series(candle_data.get('probabilities', {}))
        )
        if not trade_plan: return
        
        await self._execute_entry_workflow(trade_plan)

    async def _execute_entry_workflow(self, trade_plan: Dict[str, Any]):
        """Implements the full new entry workflow with verification."""
        logger.info(f"Executing entry workflow for a {trade_plan['direction_str']} trade.")
        entry_order, sl_order, tp_order = None, None, None
        try:
            entry_order = await self.exchange_adapter.place_market_order(
                self.symbol, trade_plan['side'], trade_plan['quantity']
            )
            sl_order, tp_order = await self.trade_execution_engine.place_and_verify_sltp_orders(trade_plan)
            liq_price = await self.exchange_adapter.get_position_liquidation_price(self.symbol)
            final_position = self.trade_execution_engine.reconcile_open_position(
                trade_plan, entry_order, sl_order, tp_order, liq_price
            )
            self.session_manager.set_open_position(final_position)
            
            # --- ADDED NOTIFICATION ---
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