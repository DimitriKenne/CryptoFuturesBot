# utils/strategy_execution/trade_execution_engine.py

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from utils.exceptions import OrderExecutionError, ConfigurationError

from config.params import AppConfig, FLOAT_EPSILON
from utils.strategy_execution.trade_calculation_helpers import TradeCalculationHelpers
from utils.exchange_adapters.exchange_interface import ExchangeInterface

logger = logging.getLogger(__name__)

class TradeExecutionEngine:
    """
    Central engine for handling all trade-related calculations and strategy logic.
    It encapsulates trade entry, exit, position sizing, SL/TP, PnL, and filtering.
    Designed for reuse in both backtesting and live trading environments,
    without direct interaction with exchange APIs or data fetching.
    """

    def __init__(self, app_config: AppConfig, symbol: str = None, exchange_adapter: Optional[ExchangeInterface] = None):
        """
        Initializes the TradeExecutionEngine by extracting all necessary configuration
        parameters from the provided AppConfig object.

        Args:
            app_config (AppConfig): The global application configuration object,
                                    containing all sub-configurations (trading, exchange, features).
                                    It is assumed that this app_config has already been
                                    validated by config/validator.py externally.
            exchange_adapter (ExchangeInterface, optional): Exchange adapter instance for live trading.
            symbol (str, optional): Trading symbol (e.g. "ADAUSDT") for live trading. Not required for backtesting.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initializing TradeExecutionEngine...")

        # --- Configuration Parsing: Extracting relevant sub-configs ---
        self.risk_config = app_config.trading.risk
        self.trade_execution_config = app_config.trading.trade_execution
        self.entry_filter_config = app_config.trading.entry_filter
        self.volatility_regime_config = app_config.trading.volatility_regime
        self.sltp_config = app_config.trading.sltp
        self.backtest_config = app_config.trading.backtest

        self.exchange_config = app_config.exchange
        self.feature_config = app_config.features

        # --- Pre-calculated Rates for Efficiency ---
        self.trading_fee_rate = self.trade_execution_config.trading_fee_pct / 100.0
        self.slippage_tolerance_rate = self.trade_execution_config.slippage_tolerance_pct / 100.0
        self.maintenance_margin_rate = self.backtest_config.maintenance_margin_pct / 100.0
        self.liquidation_fee_rate = self.backtest_config.liquidation_fee_pct / 100.0

    
        # Exchange adapter and symbol for live trading
        self.exchange_adapter = exchange_adapter
        self.symbol = symbol
        
        if self.exchange_adapter:
            # Live trading
            self.get_symbol_params = lambda: {
                "price_precision": self.exchange_adapter.price_precision,
                "quantity_precision": self.exchange_adapter.quantity_precision,
                "min_quantity": self.exchange_adapter.min_quantity,
                "min_notional": self.exchange_adapter.min_notional,
            }
        else:
            # Backtesting
            self.get_symbol_params = lambda: app_config.exchange.get_symbol_params(self.symbol)

        # --- Initialize TradeCalculationHelpers ---
        self.trade_calculation_helpers = TradeCalculationHelpers(app_config=app_config, get_symbol_params_func=self.get_symbol_params)

        self.volatility_regime_col_name = self.trade_calculation_helpers.volatility_regime_col_name
        self.atr_vol_adj_col_name = self.trade_calculation_helpers.atr_vol_adj_col_name

        self.logger.info("TradeExecutionEngine initialized with configurations and helpers.")
        

    # ====================================================================
    # --- Public API for Trade Management (Called by Backtester/TradingBot) ---
    # ====================================================================

    def calculate_entry_details(
        self,
        signal: int,
        current_capital: float,
        current_price: float,
        current_bar_features: pd.Series,
        model_probabilities: Optional[pd.Series] = None,
        current_bar_index: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Calculates all necessary details for a potential trade entry.
        This includes applying entry filters, determining position size,
        calculating Stop Loss (SL) and Take Profit (TP) prices, and
        estimating the liquidation price.

        Returns None if the trade does not pass filters or cannot be afforded.
        """
        self.logger.debug(f"Attempting to calculate entry for signal {signal} at {current_price:.{self.get_symbol_params()['price_precision']}f} on bar index {current_bar_index}")

        # 1. Input Validation: Basic sanity checks
        if signal == 0:
            self.logger.debug(f"Signal is 0 (neutral). No entry calculation needed.")
            return None
        if pd.isna(current_capital) or current_capital <= FLOAT_EPSILON:
            self.logger.warning(f"Invalid current capital ({current_capital}). Cannot calculate entry.")
            return None
        if pd.isna(current_price) or current_price <= FLOAT_EPSILON:
            self.logger.warning(f"Invalid current price ({current_price}). Cannot calculate entry.")
            return None
        if current_bar_features.empty:
            self.logger.warning("Current bar features are empty. Cannot calculate entry.")
            return None

        side = 'buy' if signal == 1 else 'sell'
        direction_str = 'long' if signal == 1 else 'short'

        filtered_signal = self.trade_calculation_helpers.apply_entry_filters(
            signal=signal,
            latest_features=current_bar_features,
            latest_probabilities=model_probabilities
        )

        if filtered_signal == 0:
            self.logger.info(f"Signal {signal} for {current_bar_features.name} was filtered out. No entry.")
            return None

        slippage_adjusted_price = current_price * (1 + self.slippage_tolerance_rate * filtered_signal)
        adjusted_entry_price = self.trade_calculation_helpers._round_price(slippage_adjusted_price)

        if pd.isna(adjusted_entry_price) or adjusted_entry_price <= FLOAT_EPSILON:
            self.logger.error(f"Adjusted entry price invalid ({adjusted_entry_price}). Cannot proceed with entry.")
            return None

        latest_atr = current_bar_features.get(self.atr_vol_adj_col_name)
        stop_loss_price, take_profit_price = self.trade_calculation_helpers.calculate_sl_tp_prices(
            side=side,
            current_price=adjusted_entry_price,
            latest_atr=latest_atr
        )

        if pd.isna(stop_loss_price) or stop_loss_price <= FLOAT_EPSILON:
            self.logger.warning(f"Stop loss price could not be calculated or is invalid ({stop_loss_price}). Blocking entry.")
            return None

        liquidation_price = self.trade_calculation_helpers.estimate_liquidation_price(
            side=side,
            entry_price=adjusted_entry_price
        )

        if pd.isna(liquidation_price) or liquidation_price <= FLOAT_EPSILON:
            self.logger.warning(f"Liquidation price could not be estimated or is invalid ({liquidation_price}). Blocking entry.")
            return None

        is_sl_safe = self.trade_calculation_helpers.is_sl_safe_from_liquidation(
            side=side,
            stop_loss_price=stop_loss_price,
            liquidation_price=liquidation_price
        )
        if not is_sl_safe:
            self.logger.warning(f"Stop loss ({stop_loss_price:.{self.get_symbol_params()['price_precision']}f}) is too close to liquidation price ({liquidation_price:.{self.get_symbol_params()['price_precision']}f}). Blocking entry.")
            return None

        adjusted_quantity, notional_value = self.trade_calculation_helpers.calculate_position_size(
            current_equity=current_capital,
            current_price=adjusted_entry_price,
            stop_loss_price=stop_loss_price,
            trade_direction=filtered_signal
        )

        if adjusted_quantity is None or adjusted_quantity <= FLOAT_EPSILON:
            self.logger.warning(f"Position size could not be determined or is zero ({adjusted_quantity}). Blocking entry.")
            return None

        initial_margin = notional_value / self.risk_config.leverage
        entry_fee = notional_value * self.trading_fee_rate

        current_regime = current_bar_features.get(self.volatility_regime_col_name, 0)
        if pd.isna(current_regime):
            self.logger.warning(f"Volatility regime for current bar is NaN. Defaulting max_holding_bars to 0 (no time limit).")
            max_holding_bars = 0
        else:
            try:
                current_regime_int = int(current_regime)
                max_holding_bars = self.volatility_regime_config.max_holding_bars.get(current_regime_int, 0)
                if max_holding_bars is None:
                    max_holding_bars = 0
                    self.logger.warning(f"Max holding bars not configured for regime {current_regime_int}. Defaulting to 0.")
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid volatility regime value '{current_regime}'. Defaulting max_holding_bars to 0.")
                max_holding_bars = 0

        entry_details = {
            'symbol': self.symbol,  # will be None if not provided (backtest)
            'direction_int': filtered_signal,
            'direction_str': direction_str,
            'side': 'BUY' if direction_str == 'long' else 'SELL',
            'entry_price': adjusted_entry_price,
            'quantity': adjusted_quantity,
            'notional_value': notional_value,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'liquidation_price': liquidation_price,
            'initial_margin': initial_margin,
            'entry_fee': entry_fee,
            'max_holding_bars': max_holding_bars,
            'entry_time': current_bar_features.name,
            'entry_bar_index': current_bar_index,
            'model_probabilities': model_probabilities.to_dict() if model_probabilities is not None else {},
            'entry_reason': 'ML_signal_entry',
        }

        self.logger.info(
            f"🟢 ENTRY CALC | {direction_str.upper()} @ {adjusted_entry_price:.4f} | Qty: {adjusted_quantity:.2f} | SL: {stop_loss_price:.4f} | TP: {take_profit_price:.4f} | Margin: {initial_margin:.2f} | Fee: {entry_fee:.2f}"
        )
        self.logger.info(f"🟢 Entry details for {direction_str} trade @ {adjusted_entry_price:.4f} | Qty: {adjusted_quantity:.2f}")
        return entry_details

    def check_exit_conditions(
        self,
        open_trade: Dict[str, Any],
        current_bar_data: pd.Series,
        current_bar_index: int
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Checks if any 'hard' or strategic exit condition (Stop Loss, Take Profit, Liquidation,
        Max Holding Period, or a Filtered Reversal Signal) is met for an open trade
        within the price range or context of the current bar.

        Returns:
            Tuple[bool, Optional[str], Optional[float]]:
                - True if an exit condition is met, False otherwise.
                - The reason for exit ('stop_loss', 'take_profit', 'liquidation', 'max_holding',
                  'reversal_signal', 'invalid_ohlc').
                - The determined exit price.
        """
        self.logger.debug(f"🔍 Checking exit conditions for {open_trade.get('direction_str').upper()} | Entry: {open_trade.get('entry_price'):.4f} | SL: {open_trade.get('stop_loss_price'):.4f} | TP: {open_trade.get('take_profit_price'):.4f} | Bar: {current_bar_index}")

        required_ohlc = ['open', 'high', 'low', 'close']
        for col in required_ohlc:
            if col not in current_bar_data.index or pd.isna(current_bar_data[col]):
                self.logger.error(f"Current bar data missing or invalid OHLC value for '{col}'. Exiting trade.")
                return True, 'invalid_ohlc', np.nan

        trade_direction_int = open_trade.get('direction_int')
        sl_price = open_trade.get('stop_loss_price')
        tp_price = open_trade.get('take_profit_price')
        liq_price = open_trade.get('liquidation_price')

        current_open = current_bar_data['open']
        current_high = current_bar_data['high']
        current_low = current_bar_data['low']
        current_close = current_bar_data['close']

        exit_price_candidate = np.nan

        # --- 1. Liquidation Check ---
        if pd.notna(liq_price) and liq_price > FLOAT_EPSILON:
            if trade_direction_int == 1:
                if current_low <= liq_price + FLOAT_EPSILON:
                    self.logger.warning(f"Long position liquidated at {liq_price:.{self.get_symbol_params()['price_precision']}f} (current_low: {current_low:.{self.get_symbol_params()['price_precision']}f}).")
                    exit_price_candidate = liq_price
                    return True, 'liquidation', self.trade_calculation_helpers._round_price(exit_price_candidate)
            elif trade_direction_int == -1:
                if current_high >= liq_price - FLOAT_EPSILON:
                    self.logger.warning(f"Short position liquidated at {liq_price:.{self.get_symbol_params()['price_precision']}f} (current_high: {current_high:.{self.get_symbol_params()['price_precision']}f}).")
                    exit_price_candidate = liq_price
                    return True, 'liquidation', self.trade_calculation_helpers._round_price(exit_price_candidate)

        # --- 2. Stop Loss (SL) Hit Check ---
        if pd.notna(sl_price) and sl_price > FLOAT_EPSILON:
            if trade_direction_int == 1:
                if current_low <= sl_price + FLOAT_EPSILON:
                    self.logger.info(f"Long position Stop Loss hit at {sl_price:.{self.get_symbol_params()['price_precision']}f} (current_low: {current_low:.{self.get_symbol_params()['price_precision']}f}).")
                    exit_price_candidate = sl_price
                    return True, 'stop_loss', self.trade_calculation_helpers._round_price(exit_price_candidate)
            elif trade_direction_int == -1:
                if current_high >= sl_price - FLOAT_EPSILON:
                    self.logger.info(f"Short position Stop Loss hit at {sl_price:.{self.get_symbol_params()['price_precision']}f} (current_high: {current_high:.{self.get_symbol_params()['price_precision']}f}).")
                    exit_price_candidate = sl_price
                    return True, 'stop_loss', self.trade_calculation_helpers._round_price(exit_price_candidate)

        # --- 3. Take Profit (TP) Hit Check ---
        if pd.notna(tp_price) and tp_price > FLOAT_EPSILON:
            if trade_direction_int == 1:
                if current_high >= tp_price - FLOAT_EPSILON:
                    self.logger.info(f"Long position Take Profit hit at {tp_price:.{self.get_symbol_params()['price_precision']}f} (current_high: {current_high:.{self.get_symbol_params()['price_precision']}f}).")
                    exit_price_candidate = tp_price
                    return True, 'take_profit', self.trade_calculation_helpers._round_price(exit_price_candidate)
            elif trade_direction_int == -1:
                if current_low <= tp_price + FLOAT_EPSILON:
                    self.logger.info(f"Short position Take Profit hit at {tp_price:.{self.get_symbol_params()['price_precision']}f} (current_low: {current_low:.{self.get_symbol_params()['price_precision']}f}).")
                    exit_price_candidate = tp_price
                    return True, 'take_profit', self.trade_calculation_helpers._round_price(exit_price_candidate)

        # --- 4. Max Holding Period Reached ---
        max_holding_bars = open_trade.get('max_holding_bars', 0)
        entry_bar_index = open_trade.get('entry_bar_index')

        if max_holding_bars > 0 and entry_bar_index is not None:
            if current_bar_index - entry_bar_index >= max_holding_bars:
                self.logger.info(f"Max holding period of {max_holding_bars} bars reached. Exiting trade.")
                exit_price_candidate = current_close
                return True, 'max_holding', self.trade_calculation_helpers._round_price(exit_price_candidate)

        # --- 5. Filtered Reversal Signal ---
        current_bar_signal = current_bar_data.get('signal')
        if current_bar_signal is not None and current_bar_signal != 0:
            if (trade_direction_int == 1 and current_bar_signal == -1) or \
               (trade_direction_int == -1 and current_bar_signal == 1):
                self.logger.info(f"Reversal signal ({current_bar_signal}) detected for open {open_trade.get('direction_str')} position. Exiting trade.")
                exit_price_candidate = current_close
                return True, 'reversal_signal', self.trade_calculation_helpers._round_price(exit_price_candidate)

        self.logger.debug(f"No exit conditions met for trade {open_trade.get('direction_str')} at bar index {current_bar_index}.")
        return False, None, None

    def calculate_exit_details(
        self,
        open_trade: Dict[str, Any],
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        current_bar_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calculates the financial outcome of a trade closure (Gross PnL, Net PnL, fees)
        and returns a comprehensive completed trade record.
        This method finalizes the trade details for historical record-keeping.

        Args:
            open_trade (Dict[str, Any]): The details of the currently open trade.
            exit_price (float): The price at which the trade was exited.
            exit_time (datetime): The timestamp of the exit.
            exit_reason (str): The reason for the exit ('stop_loss', 'take_profit', 'liquidation',
                               'max_holding', 'reversal_signal', 'invalid_ohlc').
            current_bar_index (Optional[int]): The index of the bar at which the trade exited.
                                                Required for calculating holding duration in bars.

        Returns:
            Dict[str, Any]: A comprehensive dictionary representing the completed trade record.
        """
        self.logger.debug(f"Calculating exit details for trade (entry {open_trade.get('entry_price')}) at exit price {exit_price} due to {exit_reason}")

        if not open_trade or pd.isna(exit_price) or exit_price <= FLOAT_EPSILON or not isinstance(exit_time, datetime):
            self.logger.error("Invalid input for calculate_exit_details. Cannot calculate exit details.")
            return {}

        actual_exit_price = exit_price
        if exit_reason != 'liquidation':
            direction_int = open_trade.get('direction_int', 0)
            slippage_multiplier = -1 if direction_int == 1 else (1 if direction_int == -1 else 0)
            actual_exit_price = exit_price * (1 + self.slippage_tolerance_rate * slippage_multiplier)
            actual_exit_price = self.trade_calculation_helpers._round_price(actual_exit_price)
            if pd.isna(actual_exit_price) or actual_exit_price <= FLOAT_EPSILON:
                self.logger.error(f"Actual exit price invalid ({actual_exit_price}) after slippage adjustment. Using original exit price.")
                actual_exit_price = exit_price

        quantity = open_trade.get('quantity', 0.0)
        notional_value_at_exit = actual_exit_price * quantity

        gross_pnl, exit_fee, liquidation_fee, net_pnl = \
            self.trade_calculation_helpers.calculate_pnl_and_fees(
                trade_direction_int=open_trade.get('direction_int', 0),
                entry_price=open_trade.get('entry_price', 0.0),
                actual_exit_price=actual_exit_price,
                quantity=quantity,
                entry_fee=open_trade.get('entry_fee', 0.0),
                trading_fee_rate=self.trading_fee_rate,
                liquidation_fee_rate=self.liquidation_fee_rate,
                notional_value_at_exit=notional_value_at_exit,
                exit_reason=exit_reason
            )

        holding_bars = None
        if current_bar_index is not None and open_trade.get('entry_bar_index') is not None:
            holding_bars = current_bar_index - open_trade['entry_bar_index']
            if holding_bars < 0:
                self.logger.warning(f"Calculated holding_bars is negative ({holding_bars}). Setting to 0.")
                holding_bars = 0

        holding_duration_seconds = 0.0
        if isinstance(open_trade.get('entry_time'), datetime) and isinstance(exit_time, datetime):
            holding_duration_seconds = (exit_time - open_trade['entry_time']).total_seconds()
            if holding_duration_seconds < 0:
                self.logger.warning(f"Calculated holding_duration_seconds is negative ({holding_duration_seconds}). Setting to 0.")
                holding_duration_seconds = 0.0

        completed_trade = open_trade.copy()
        completed_trade.update({
            'exit_price': actual_exit_price,
            'exit_time': exit_time,
            'exit_reason': exit_reason,
            'gross_pnl': gross_pnl,
            'exit_fee': exit_fee,
            'liquidation_fee': liquidation_fee,
            'total_fees': open_trade.get('entry_fee', 0.0) + exit_fee + liquidation_fee,
            'net_pnl': net_pnl,
            'holding_bars': holding_bars,
            'holding_duration_seconds': holding_duration_seconds,
            'is_closed': True,
            'notional_value_at_exit': notional_value_at_exit
        })

        self.logger.info(
            f"🔴 EXIT CALC | {open_trade.get('direction_str','?').upper()} Entry @ {open_trade.get('entry_price',0):.4f} | Exit @ {actual_exit_price:.4f} | NetPnL: {net_pnl:.2f} | Reason: {exit_reason} | Fees: {exit_fee:.2f}+{liquidation_fee:.2f}"
        )
        self.logger.info(f"🔴 Trade closed due to '{exit_reason}' @ {actual_exit_price:.4f} | NetPnL: {net_pnl:.2f}")
        return completed_trade

    # ====================================================================
    # --- Methods for Live Trading Workflow ---
    # ====================================================================

    async def cancel_remaining_orders(self, position: Dict[str, Any], exit_reason: str):
        """Orchestrates cancellation of the non-triggered SL/TP order."""
        if self.exchange_adapter is None:
            raise ConfigurationError("Exchange adapter is not configured in TradeExecutionEngine.")

        sl_order_id = position.get('sl_order_id')
        tp_order_id = position.get('tp_order_id')
        symbol = position.get('symbol', self.symbol)

        order_to_cancel = None
        if exit_reason == 'stop_loss' and tp_order_id:
            order_to_cancel = tp_order_id
        elif exit_reason == 'take_profit' and sl_order_id:
            order_to_cancel = sl_order_id
        elif sl_order_id and tp_order_id:
            await self.exchange_adapter.cancel_multiple_orders(symbol, [sl_order_id, tp_order_id])
            return

        if order_to_cancel and symbol:
            await self.exchange_adapter.cancel_order(symbol, order_to_cancel)

    def reconcile_open_position(self, trade_plan: Dict, entry_confirmation: Dict, sl_order: Dict, tp_order: Dict, liq_price: float) -> Dict[str, Any]:
        """Creates the final, verified position dictionary using real data from the exchange."""
        final_position = trade_plan.copy()
        final_position.update({
            'entry_price': entry_confirmation['avgPrice'],
            'quantity': entry_confirmation['executedQty'],
            'entry_time': entry_confirmation['time'],
            'entry_order_id': entry_confirmation['orderId'],
            'sl_order_id': sl_order['orderId'],
            'tp_order_id': tp_order['orderId'],
            'liquidation_price': liq_price,
            'id': str(entry_confirmation['orderId']) 
        })
        self.logger.info(f"Position reconciled with exchange data: ID {final_position.get('id', '?')}")
        return final_position

    async def place_and_verify_sltp_orders(self, trade_plan: Dict) -> Tuple[Dict, Dict]:
        """Places and robustly verifies SL and TP orders, returning confirmed order details.

        Requires 'symbol' and 'side' to be present in trade_plan (or self.symbol for symbol).
        """
        if self.exchange_adapter is None:
            raise ConfigurationError("Exchange adapter is not configured in TradeExecutionEngine.")

        symbol = trade_plan.get('symbol', self.symbol)
        if symbol is None:
            raise ConfigurationError("Trade plan does not have a symbol and self.symbol is not set. This is required for live trading.")

        side_to_close = 'SELL' if trade_plan.get('side') == 'BUY' else 'BUY'
        quantity = trade_plan['quantity']

        sl_order = await self.exchange_adapter.place_stop_market_order(
            symbol=symbol, side=side_to_close, quantity=quantity, stop_price=trade_plan['stop_loss_price']
        )
        await self._verify_order(symbol, sl_order['orderId'])

        tp_order = await self.exchange_adapter.place_take_profit_market_order(
            symbol=symbol, side=side_to_close, quantity=quantity, stop_price=trade_plan['take_profit_price']
        )
        await self._verify_order(symbol, tp_order['orderId'])

        return sl_order, tp_order

    async def _verify_order(self, symbol: str, order_id: str, timeout: int = 10, delay: float = 0.5):
        """Continuously checks an order's status until it's confirmed or times out."""
        if self.exchange_adapter is None:
            raise ConfigurationError("Exchange adapter is not configured for order verification.")

        start_time = datetime.now()
        while (datetime.now() - start_time).total_seconds() < timeout:
            order_info = await self.exchange_adapter.get_order_info(symbol, order_id)
            if order_info and order_info.get('status') == 'NEW':
                self.logger.info(f"Order {order_id} successfully verified on exchange.")
                return
            await asyncio.sleep(delay)
        raise OrderExecutionError(f"Failed to verify order {order_id} within {timeout} seconds.")

    async def cleanup_failed_entry(self, symbol: str, entry_order: Optional[Dict], sl_order: Optional[Dict], tp_order: Optional[Dict]):
        """Attempts to clean up any residual orders or positions from a failed entry sequence."""
        if self.exchange_adapter is None: return

        self.logger.warning("--- INITIATING FAILED ENTRY CLEANUP ---")
        orders_to_cancel = [o['orderId'] for o in [sl_order, tp_order] if o and o.get('orderId')]
        if orders_to_cancel:
            await self.exchange_adapter.cancel_multiple_orders(symbol, orders_to_cancel)

        open_positions = await self.exchange_adapter.get_open_positions(symbol)
        if open_positions:
            self.logger.critical(f"A position for {symbol} exists after failed entry. Attempting to close it immediately.")
            pos_to_close = open_positions[0]
            side_to_close = 'SELL' if pos_to_close['direction'] == 'long' else 'BUY'
            await self.exchange_adapter.place_market_order(symbol, side_to_close, pos_to_close['quantity'], reduce_only=True)
        self.logger.warning("--- FAILED ENTRY CLEANUP COMPLETE ---")