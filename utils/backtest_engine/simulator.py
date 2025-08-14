import pandas as pd
import numpy as np
import math
import logging
import time
import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timezone

# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration Imports ---
# Load configurations from the central config files. Assumes standard project structure.
try:
    from config.paths import PATHS
    # Import the aggregated app_config directly
    from config.params import app_config, FLOAT_EPSILON # Import FLOAT_EPSILON here

    # Import specific configuration dataclass types for type hinting and direct access
    from config.backtest_config_schema import BacktestConfig
    from config.strategy_config_schema import StrategyConfig, VolatilityRegimeConfig, SLTPConfig
    from config.exchange_config_schema import ExchangeConfig
    from config.feature_config_schema import FeatureConfig # To get ATR period for volatility regime column name
except ImportError as e:
    logging.critical(f"Failed to import configuration modules: {e}")
    raise # Re-raise the exception to stop execution if essential imports fail

# --- NEW: Import OHLCVProcessor for data preparation ---
from utils.data_processing.ohlcv_processor import OHLCVProcessor

# --- NEW IMPORTS for trade_core ---
from utils.trade_core.financial_math import FinancialMath
from utils.trade_core.position_sizing import PositionSizer
from utils.trade_core.liquidation import LiquidationEstimator
from utils.trade_core.order_precision import OrderPrecisionHandler

# --- NEW IMPORTS for strategy_execution ---
from utils.strategy_execution.trade_manager import TradeManager
from utils.strategy_execution.entry_filters import EntryFilters
from utils.strategy_execution.exit_conditions import ExitConditions


logger = logging.getLogger(__name__)

class BacktestSimulator:
    """
    Simulates trading operations based on historical OHLCV data, model signals,
    and defined strategy rules. It processes data bar by bar, manages trades,
    applies risk management, and calculates performance metrics.

    This is a revised, more robust backtesting engine that integrates modular
    trade execution and filtering logic.
    """

    def __init__(self,
                 data: pd.DataFrame,
                 model_predict: pd.Series,
                 model_proba: pd.DataFrame,
                 symbol: str,
                 interval: str,
                 model_type: str,
                 paths_override: Optional[Dict[str, Union[str, Path]]] = None # To allow custom path during testing
                 ):
        """
        Initializes the BacktestSimulator with historical data, model predictions,
        and configuration settings.

        Args:
            data (pd.DataFrame): Historical OHLCV data with engineered features.
            model_predict (pd.Series): Series with model predictions (-1, 0, 1) aligned with data index.
            model_proba (pd.DataFrame): DataFrame of model probabilities for each class (-1, 0, 1),
                                        aligned with data index.
            symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
            interval (str): OHLCV interval (e.g., '1h', '5m').
            model_type (str): Type of model used (e.g., 'xgboost', 'random_forest').
            paths_override (Optional[Dict]): Dictionary to override default paths from config.paths.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"--- Initializing BacktestSimulator for {symbol} {interval} ({model_type}) ---")

        # --- Core Data Inputs ---
        self.raw_data = data.copy() # Keep a copy of original data for reference
        self.model_predict = model_predict.copy()
        self.model_proba = model_proba.copy()

        # --- Configuration from app_config ---
        # Access config directly from the imported app_config object
        self.strategy_config: StrategyConfig = app_config.strategy
        self.backtest_config: BacktestConfig = app_config.backtest
        self.exchange_config: ExchangeConfig = app_config.exchange
        self.feature_config: FeatureConfig = app_config.features # Access FeatureConfig

        # Apply strategy overrides from backtest config
        self._apply_strategy_overrides()

        # --- Backtest Specific Parameters ---
        self.initial_capital = self.strategy_config.initial_capital
        self.capital = self.initial_capital
        self.max_concurrent_trades = self.backtest_config.max_concurrent_trades
        self.maintenance_margin_rate = self.backtest_config.maintenance_margin_rate
        self.liquidation_fee_rate = self.backtest_config.liquidation_fee_rate

        self.symbol = symbol
        self.interval = interval
        self.model_type = model_type

        # --- Data Buffers and Pointers ---
        self.current_position_direction = 0 # 1: Long, -1: Short, 0: Flat
        self.open_trade = None # Stores a dictionary of current trade details if position is open
        self.trade_history = [] # List to store completed trade records

        # --- Performance Tracking ---
        self.equity_curve = pd.Series(dtype=float, name='equity') # Stores capital at each bar close

        # --- Price and Quantity Precision from ExchangeConfig ---
        # NOTE: In a real scenario, these would be fetched live from exchange info
        # For backtesting, we assume static values from config
        self.price_precision = self.exchange_config.price_precision
        self.quantity_precision = self.exchange_config.quantity_precision
        self.min_quantity = self.exchange_config.min_quantity
        self.min_notional = self.exchange_config.min_notional
        
        # Log a warning if any are still None (shouldn't happen if __post_init__ is correctly validating defaults)
        if self.price_precision is None:
            self.logger.warning("Price precision not found in exchange config, defaulting to 4 for methods.")
            self.price_precision = 4
        if self.quantity_precision is None:
            self.logger.warning("Quantity precision not found in exchange config, defaulting to 8 for methods.")
            self.quantity_precision = 8
        if self.min_quantity is None:
            self.logger.warning("Min quantity not found in exchange config, defaulting to 0.001 for methods.")
            self.min_quantity = 0.001
        if self.min_notional is None:
            self.logger.warning("Min notional not found in exchange config, defaulting to 10.0 for methods.")
            self.min_notional = 10.0


        # --- Path Configuration (can be overridden for specific test runs) ---
        self.paths = PATHS.copy()
        if paths_override:
            self.paths.update(paths_override)

        # --- Initialize Utility Classes ---
        self.ohlcv_processor = OHLCVProcessor()
        self.financial_math = FinancialMath()
        self.position_sizer = PositionSizer()
        self.liquidation_estimator = LiquidationEstimator()
        self.order_precision_handler = OrderPrecisionHandler(
            price_precision=self.price_precision,
            quantity_precision=self.quantity_precision,
            min_quantity=self.min_quantity,
            min_notional=self.min_notional
        )

        # Initialize TradeManager, EntryFilters, ExitConditions
        self.trade_manager = TradeManager(
            strategy_config=self.strategy_config,
            exchange_config=self.exchange_config,
            backtest_config=self.backtest_config,
            symbol=self.symbol,
            interval=self.interval,
            model_type=self.model_type
        )

        self.entry_filters = EntryFilters(
            confidence_filter_enabled=self.strategy_config.confidence_filter_enabled,
            confidence_threshold_long=self.strategy_config.confidence_threshold_long_pct,
            confidence_threshold_short=self.strategy_config.confidence_threshold_short_pct,
            volatility_regime_filter_enabled=self.strategy_config.volatility_regime_filter_enabled,
            volatility_regime_col_name=self.feature_config.volatility_regime_col_name,
            allow_trading_in_volatility_regime=self.strategy_config.volatility_regime_params.allow_trading,
            trend_filter_enabled=self.strategy_config.trend_filter_enabled,
            ema_filter_col_name=f"ema_{self.strategy_config.trend_filter_ema_period}", # Ensure this is a feature column
            price_precision=self.price_precision
        )

        self.exit_conditions = ExitConditions(
            strategy_config=self.strategy_config,
            backtest_config=self.backtest_config,
            exchange_config=self.exchange_config 
        )

        # --- Flags for saving results ---
        self.save_trades = self.backtest_config.save_trades
        self.save_equity_curve = self.backtest_config.save_equity_curve
        self.save_metrics = self.backtest_config.save_metrics

        self.logger.info("BacktestSimulator initialized successfully.")


    def _apply_strategy_overrides(self):
        """Applies overrides from backtest_config.override_strategy_params to strategy_config."""
        if self.backtest_config.override_strategy_params:
            self.logger.info("Applying strategy parameter overrides for backtesting...")
            for param, value in self.backtest_config.override_strategy_params.items():
                if hasattr(self.strategy_config, param):
                    original_value = getattr(self.strategy_config, param)
                    # Use object.__setattr__ to modify frozen dataclass
                    object.__setattr__(self.strategy_config, param, value)
                    self.logger.info(f"  Overrode '{param}': {original_value} -> {value}")
                else:
                    self.logger.warning(f"  Attempted to override non-existent strategy parameter: '{param}'")

    def _prepare_data(self) -> pd.DataFrame:
        """
        Prepares the raw data for simulation by merging with predictions and probabilities,
        and ensuring all necessary columns are present and correctly formatted.
        """
        # Ensure indices match for merging
        if not self.raw_data.index.equals(self.model_predict.index):
            self.logger.warning("Data and prediction indices do not match. Reindexing predictions.")
            self.model_predict = self.model_predict.reindex(self.raw_data.index).fillna(0).astype(int)

        if not self.raw_data.index.equals(self.model_proba.index):
            self.logger.warning("Data and probability indices do not match. Reindexing probabilities.")
            model_proba_reindexed = self.model_proba.reindex(self.raw_data.index) # Use 'self.raw_data.index' here
            # Ensure probability columns exist after reindexing if they were missing
            for col in [-1, 0, 1]:
                if col not in model_proba_reindexed.columns:
                    model_proba_reindexed[col] = np.nan
            self.model_proba = model_proba_reindexed # Update self.model_proba
            

        # Use the OHLCVProcessor to prepare and combine data
        processed_data = self.ohlcv_processor.prepare_data(
            data=self.raw_data,
            model_predict=self.model_predict,
            model_proba=self.model_proba,
            # Pass individual parameters extracted from config objects
            volatility_adjustment_enabled=self.strategy_config.sltp_params.enabled, # Use the SLTP enabled flag for volatility adjustment
            trend_filter_enabled=self.strategy_config.trend_filter_enabled,
            volatility_regime_filter_enabled=self.strategy_config.volatility_regime_filter_enabled,
            atr_col_name=f"atr_{self.strategy_config.sltp_params.volatility_window_bars}", # Dynamic ATR column name
            ema_col_name=f"ema_{self.strategy_config.trend_filter_ema_period}", # Dynamic EMA column name
            volatility_regime_col_name=self.feature_config.volatility_regime_col_name
        )

        if processed_data.empty:
            self.logger.error("Prepared data is empty after processing. Cannot run backtest.")
            raise ValueError("Prepared data is empty.")

        # Ensure that the 'signal' and 'proba_-1', 'proba_0', 'proba_1' columns exist
        if 'signal' not in processed_data.columns:
            self.logger.critical("Prepared data is missing the 'signal' column after OHLCV processing.")
            raise ValueError("Missing 'signal' column in prepared data.")
        
        # Check for expected probability columns (as strings as they might be column names)
        for prob_col in ['proba_-1', 'proba_0', 'proba_1']:
            if prob_col not in processed_data.columns:
                self.logger.warning(f"Probability column '{prob_col}' missing in prepared data. Confidence filters may not function.")
                processed_data[prob_col] = np.nan # Add it filled with NaN to prevent downstream errors

        self.logger.info(f"Data prepared for backtest. Shape: {processed_data.shape}")
        return processed_data


    def run_backtest(self) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        """
        Executes the backtest simulation bar by bar.

        Returns:
            Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
                - Detailed trade history.
                - Equity curve over time.
                - Summary performance metrics.
        """
        self.logger.info(f"--- Starting Backtest Simulation ({len(self.raw_data)} bars) ---")

        # Prepare data (merge predictions, add indicators etc.)
        try:
            processed_data = self._prepare_data()
        except ValueError as e:
            self.logger.critical(f"Data preparation failed: {e}. Aborting backtest.")
            return pd.DataFrame(), pd.Series(dtype=float), {} # Return empty results

        # Initialize equity curve with initial capital at the start of data
        first_timestamp = processed_data.index[0]
        self.equity_curve = pd.Series(self.capital, index=[first_timestamp], dtype=float)


        # Main backtest loop
        # We iterate over the *index* of the processed data to ensure chronological order
        # and access data using .loc[timestamp]
        for i, (timestamp, current_bar) in enumerate(processed_data.iterrows()):
            # Update the current capital for the trade manager
            # self.trade_manager.current_capital = self.capital # This is not needed if TradeManager is passed configs

            current_close_price = current_bar['close']
            current_signal = int(current_bar['signal']) # Ensure signal is integer
            
            # Get probability scores for the current bar
            current_long_proba = current_bar.get('proba_1', np.nan)
            current_short_proba = current_bar.get('proba_-1', np.nan)
            
            # Update TradeManager with current bar info (for max holding, ATR for SL/TP etc.)
            # TradeManager needs `current_bar` to derive `atr` and `volatility_regime`
            self.trade_manager.update_current_bar_info(current_bar)


            # --- 1. Check for exiting an open trade ---
            if self.open_trade:
                exit_decision, exit_reason, exit_price = self.exit_conditions.check_conditions( # Renamed method
                    current_bar=current_bar,
                    position_direction=self.open_trade['direction_int'], # Assuming direction_int is stored in open_trade
                    current_trade_sl_price=self.open_trade['sl_price'],
                    current_trade_tp_price=self.open_trade['tp_price'],
                    trade_open_bar_index=self.open_trade['trade_open_bar_index'],
                    current_bar_index=i, # Pass current bar index
                    trade_max_holding_bars=self.open_trade['trade_max_holding_bars'],
                    liquidation_price=self.open_trade['liquidation_price'],
                    price_precision=self.price_precision # Pass price precision
                )

                if exit_decision:
                    # Execute exit and record trade
                    trade_record = self.trade_manager.close_trade(
                        open_trade=self.open_trade,
                        exit_price=exit_price,
                        exit_time=timestamp,
                        exit_reason=exit_reason,
                        current_capital=self.capital, # Pass current capital
                        price_precision=self.price_precision,
                        quantity_precision=self.quantity_precision
                    )
                    if trade_record:
                        # PnL and fees are now part of the trade_record from close_trade
                        pnl = trade_record['net_pnl']
                        # fees = trade_record['total_fees'] # No longer needed here, PnL is net
                        self.capital += pnl # Update capital with net PnL
                        self.trade_history.append(trade_record)
                        self.open_trade = None # Close the position
                        self.logger.info(f"Trade closed at {timestamp} due to {exit_reason}. Net PnL: {pnl:.2f}, New Capital: {self.capital:.2f}")
                    else:
                        self.logger.error(f"Failed to close trade for reason {exit_reason} at {timestamp}.")
                else:
                    # Update open trade with current market data for correct PnL tracking
                    # The return value is used directly in the equity calculation below
                    # We don't need to assign it to open_trade here, as open_trade is not meant
                    # to hold real-time unrealized PnL within its dict.
                    self.trade_manager.calculate_unrealized_pnl(current_bar['close'], self.open_trade) 
            
            # --- 2. Check for opening a new trade (only if no open trade) ---
            if not self.open_trade:
                # Apply entry filters FIRST
                if self.entry_filters.apply_filters(
                    current_signal,
                    current_bar,
                    current_long_proba, # Pass the specific probability for long
                    current_short_proba, # Pass the specific probability for short
                    self.price_precision # Pass price precision for logging
                ):
                    # Check if signal allows for a new trade given current open/short/long config
                    if current_signal == 1 and self.strategy_config.allow_long_trades:
                        if self.max_concurrent_trades > len([t for t in self.trade_history if t.get('status') == 'OPEN']): # Check actual open trades
                             new_trade = self.trade_manager.open_trade(
                                 signal=1,
                                 current_capital=self.capital,
                                 current_bar_data=current_bar,
                                 bar_index=i, # Pass current bar index
                             )
                             if new_trade:
                                 self.open_trade = new_trade
                                 # Capital is reduced by initial margin, but it's part of PnL calculation in TradeManager
                                 # So we don't adjust capital here for entry fee, it's handled on close.
                                 self.logger.info(f"Opened LONG position at {timestamp} @ {new_trade['entry_price']:.{self.price_precision}f} with {new_trade['quantity']:.{self.quantity_precision}f} quantity.")
                                 # Add direction_int to open_trade for consistency with ExitConditions
                                 self.open_trade['direction_int'] = 1
                             else:
                                 self.logger.warning(f"Could not open LONG position at {timestamp}. TradeManager returned None.")
                        else:
                            self.logger.debug(f"Skipping LONG signal at {timestamp}: Max concurrent trades reached.")

                    elif current_signal == -1 and self.strategy_config.allow_short_trades:
                        if self.max_concurrent_trades > len([t for t in self.trade_history if t.get('status') == 'OPEN']): # Check actual open trades
                             new_trade = self.trade_manager.open_trade(
                                 signal=-1,
                                 current_capital=self.capital,
                                 current_bar_data=current_bar,
                                 bar_index=i, # Pass current bar index
                             )
                             if new_trade:
                                 self.open_trade = new_trade
                                 # Capital is reduced by initial margin, but it's part of PnL calculation in TradeManager
                                 # So we don't adjust capital here for entry fee, it's handled on close.
                                 self.logger.info(f"Opened SHORT position at {timestamp} @ {new_trade['entry_price']:.{self.price_precision}f} with {new_trade['quantity']:.{self.quantity_precision}f} quantity.")
                                 # Add direction_int to open_trade for consistency with ExitConditions
                                 self.open_trade['direction_int'] = -1
                             else:
                                 self.logger.warning(f"Could not open SHORT position at {timestamp}. TradeManager returned None.")
                        else:
                            self.logger.debug(f"Skipping SHORT signal at {timestamp}: Max concurrent trades reached.")
                else:
                    self.logger.debug(f"Trade signal {current_signal} at {timestamp} blocked by entry filters.")


            # --- Record equity at each bar close ---
            # If there's an open trade, calculate current PnL and add to capital
            current_equity = self.capital
            if self.open_trade:
                # Use the current bar's close price for mark-to-market PnL
                current_equity = self.capital + self.trade_manager.calculate_unrealized_pnl(
                    current_bar['close'], self.open_trade
                )
            
            self.equity_curve.loc[timestamp] = current_equity
        
        # After loop, if there's an open trade, close it at the last price for final calculation
        if self.open_trade:
            last_bar_close = processed_data['close'].iloc[-1]
            last_timestamp = processed_data.index[-1]
            self.logger.info(f"Closing remaining open position at end of backtest ({last_timestamp}) at last close price: {last_bar_close:.{self.price_precision}f}")
            trade_record = self.trade_manager.close_trade(
                open_trade=self.open_trade,
                exit_price=last_bar_close,
                exit_time=last_timestamp,
                exit_reason='end_of_backtest',
                current_capital=self.capital,
                price_precision=self.price_precision,
                quantity_precision=self.quantity_precision
            )
            if trade_record:
                pnl = trade_record['net_pnl']
                # fees = trade_record['total_fees'] # Not needed here
                self.capital += pnl # Update capital with net PnL
                self.trade_history.append(trade_record)
                self.equity_curve.loc[last_timestamp] = self.capital # Update final equity
                self.open_trade = None
            else:
                self.logger.error(f"Failed to close final trade at {last_timestamp}.")

        self.logger.info("--- Backtest Simulation Complete ---")

        trades_df = pd.DataFrame(self.trade_history)
        if not trades_df.empty:
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            trades_df.set_index('entry_time', inplace=True)
            self.logger.info(f"Total trades recorded: {len(trades_df)}")
        else:
            self.logger.warning("No trades were executed during the backtest.")

        # Ensure equity curve index is DatetimeIndex and sorted
        self.equity_curve.index = pd.to_datetime(self.equity_curve.index)
        self.equity_curve = self.equity_curve.sort_index()

        summary_metrics = self._calculate_summary_metrics(trades_df, self.equity_curve)

        # Save results based on flags
        if self.save_trades:
            self._save_trades(trades_df)
        if self.save_equity_curve:
            self._save_equity_curve(self.equity_curve)
        if self.save_metrics:
            self._save_metrics(summary_metrics)

        return trades_df, self.equity_curve, summary_metrics


    def _calculate_summary_metrics(self, trades_df: pd.DataFrame, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculates key performance metrics for the backtest."""
        metrics = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'net_profit': self.capital - self.initial_capital,
            'return_on_capital_pct': ((self.capital - self.initial_capital) / self.initial_capital) * 100.0 if self.initial_capital > FLOAT_EPSILON else 0.0,
            'max_drawdown': 0.0,
            'total_trades': len(trades_df),
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'total_profit_trades': 0,
            'total_loss_trades': 0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'avg_profit_per_trade': 0.0,
            'avg_loss_per_trade': 0.0,
            'avg_pnl_per_trade': 0.0,
            'edge_expected_value': 0.0, # Average PnL per trade including zeros for no-trade bars
            'total_fees': trades_df['total_fees'].sum() if not trades_df.empty else 0.0,
            'avg_holding_duration_bars': 0.0,
            'equity_pnl_discrepancy': np.nan # For debugging: difference between final capital and equity curve end
        }

        if not trades_df.empty:
            profit_trades = trades_df[trades_df['net_pnl'] > FLOAT_EPSILON] # Use net_pnl for profit/loss calculation
            loss_trades = trades_df[trades_df['net_pnl'] < -FLOAT_EPSILON]

            metrics['total_profit_trades'] = len(profit_trades)
            metrics['total_loss_trades'] = len(loss_trades)
            
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = (metrics['total_profit_trades'] / metrics['total_trades']) * 100.0
                metrics['avg_pnl_per_trade'] = trades_df['net_pnl'].sum() / metrics['total_trades']

            metrics['gross_profit'] = profit_trades['net_pnl'].sum()
            metrics['gross_loss'] = loss_trades['net_pnl'].sum()

            if abs(metrics['gross_loss']) > FLOAT_EPSILON:
                metrics['profit_factor'] = metrics['gross_profit'] / abs(metrics['gross_loss'])
            else:
                metrics['profit_factor'] = np.inf if metrics['gross_profit'] > FLOAT_EPSILON else 0.0 # Handle division by zero

            if metrics['total_profit_trades'] > 0:
                metrics['avg_profit_per_trade'] = metrics['gross_profit'] / metrics['total_profit_trades']
            if metrics['total_loss_trades'] > 0:
                metrics['avg_loss_per_trade'] = metrics['gross_loss'] / metrics['total_loss_trades']
            
            # Calculate average holding duration
            trades_df['holding_duration'] = (trades_df['exit_time'] - trades_df.index).dt.total_seconds() / (self.trade_manager._get_interval_seconds(self.interval) or 1)
            metrics['avg_holding_duration_bars'] = trades_df['holding_duration'].mean()


        if not equity_curve.empty:
            peak = equity_curve.expanding(min_periods=1).max()
            drawdown = (equity_curve - peak) / peak
            metrics['max_drawdown'] = abs(drawdown.min()) * 100.0 # As a positive percentage
            
            # Calculate daily/intervally returns for Sharpe/Sortino
            returns = equity_curve.pct_change().dropna()
            
            # Use risk-free rate as 0 for simplicity, and scale by square root of periods per year/interval.
            # Assuming returns are per interval, we need to annualize.
            # This is a simplification; for proper annualization, ensure interval-to-year conversion.
            # For backtesting, often a per-interval Sharpe/Sortino is fine.
            # Let's assume returns are already normalized for the frequency (e.g., 5m returns)
            
            if not returns.empty and returns.std() > FLOAT_EPSILON:
                metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(app_config.strategy.bars_per_year) # Assuming bars_per_year is config
                
                downside_returns = returns[returns < 0]
                if not downside_returns.empty and downside_returns.std() > FLOAT_EPSILON:
                    metrics['sortino_ratio'] = returns.mean() / downside_returns.std() * np.sqrt(app_config.strategy.bars_per_year)
                else:
                    metrics['sortino_ratio'] = np.inf if returns.mean() > FLOAT_EPSILON else 0.0
            
            # Equity PnL Discrepancy (for debugging equity vs. trade PnL)
            metrics['equity_pnl_discrepancy'] = (self.capital - self.initial_capital) - metrics['net_profit']
            if abs(metrics['equity_pnl_discrepancy']) > FLOAT_EPSILON:
                self.logger.warning(f"Discrepancy between final capital and net profit from trades: {metrics['equity_pnl_discrepancy']:.4f}")


        self.logger.info("Summary metrics calculated. Make sure 'bars_per_year' is appropriately set in strategy_config for annualized metrics.")
        return metrics

    def _save_trades(self, trades_df: pd.DataFrame):
        """Saves the detailed trade log to a Parquet file."""
        if trades_df.empty:
            self.logger.warning("No trades to save.")
            return

        save_dir = self.paths.get('backtesting_results_dir')
        file_pattern = self.paths.get('backtesting_trades_pattern')

        if not save_dir or not file_pattern:
            self.logger.error("Trades save path or pattern not configured in paths.py. Cannot save.")
            return

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = file_pattern.format(
                symbol=self.symbol.replace('/', ''),
                interval=self.interval,
                model_type=self.model_type,
                timestamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            )
            filepath = save_dir / filename
            trades_df.to_parquet(filepath, index=True) # Save with index (entry_time)
            self.logger.info(f"Trade history saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving trades to {filepath}: {e}", exc_info=True)

    def _save_equity_curve(self, equity_curve: pd.Series):
        """Saves the equity curve to a Parquet file."""
        if equity_curve.empty:
            self.logger.warning("Equity curve is empty. Nothing to save.")
            return
        
        save_dir = self.paths.get('backtesting_results_dir')
        file_pattern = self.paths.get('backtesting_equity_pattern')

        if not save_dir or not file_pattern:
            self.logger.error("Equity curve save path or pattern not configured in paths.py. Cannot save.")
            return

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = file_pattern.format(
                symbol=self.symbol.replace('/', ''),
                interval=self.interval,
                model_type=self.model_type,
                timestamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            )
            filepath = save_dir / filename
            # Convert Series to DataFrame before saving to parquet
            equity_curve.to_frame(name='equity').to_parquet(filepath, index=True) 
            self.logger.info(f"Equity curve saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving equity curve to {filepath}: {e}", exc_info=True)

    def _save_metrics(self, metrics: Dict[str, Any]):
        """Saves the summary performance metrics to a JSON file."""
        if not metrics:
            self.logger.warning("No metrics to save.")
            return

        save_dir = self.paths.get('backtesting_results_dir')
        file_pattern = self.paths.get('backtesting_metrics_pattern')

        if not save_dir or not file_pattern:
            self.logger.error("Metrics save path or pattern not configured in paths.py. Cannot save.")
            return

        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            filename = file_pattern.format(
                symbol=self.symbol.replace('/', ''),
                interval=self.interval,
                model_type=self.model_type,
                timestamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            )
            filepath = save_dir / filename
            
            # Convert any numpy types to native Python types for JSON serialization
            serializable_metrics = {k: (v.item() if isinstance(v, (np.integer, np.floating, np.bool_)) else v)
                                    for k, v in metrics.items()}

            with open(filepath, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
            self.logger.info(f"Summary metrics saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving metrics to {filepath}: {e}", exc_info=True)
