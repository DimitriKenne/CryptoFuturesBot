# config/strategy_config_schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class VolatilityRegimeConfig:
    """Configuration for volatility regime filtering."""
    max_holding_bars: Dict[int, int] = field(default_factory=lambda: {
        0: 200, # Low volatility: Longer holding period
        1: 100, # Medium volatility: Standard holding period
        2: 50   # High volatility: Shorter holding period to reduce exposure
    })
    allow_trading: Dict[int, bool] = field(default_factory=lambda: {
        0: True,
        1: True,
        2: True # Can set to False if you want to avoid trading in high volatility
    })

    def __post_init__(self):
        if not isinstance(self.max_holding_bars, dict) or set(self.max_holding_bars.keys()) != {0, 1, 2}:
            raise ValueError("'max_holding_bars' must be a dictionary with keys 0, 1, 2.")
        if not isinstance(self.allow_trading, dict) or set(self.allow_trading.keys()) != {0, 1, 2}:
            raise ValueError("'allow_trading' must be a dictionary with keys 0, 1, 2.")
        for k, v in self.allow_trading.items():
            if not isinstance(v, bool):
                raise TypeError(f"Value for allow_trading key {k} must be a boolean.")
        for k, v in self.max_holding_bars.items():
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"Value for max_holding_bars key {k} must be a positive integer.")


@dataclass
class SLTPConfig:
    """Configuration for Stop Loss (SL) and Take Profit (TP) parameters."""
    enabled: bool = True                    # Whether to enable dynamic SL/TP based on ATR. If False, fixed values are used.
    volatility_window_bars: int = 20        # ATR period for dynamic SL/TP calculation.
    fixed_take_profit_pct: float = 2.0      # Fixed TP percentage if dynamic is disabled.
    fixed_stop_loss_pct: float = 1.0        # Fixed SL percentage if dynamic is disabled.
    alpha_take_profit: float = 1.5          # Multiplier for ATR to set TP (e.g., 1.5 * ATR).
    alpha_stop_loss: float = 2.0            # Multiplier for ATR to set SL (e.g., 2.0 * ATR).
    min_sl_tp_pct: float = 0.1              # Minimum percentage for SL/TP to prevent overly tight stops.

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise TypeError("SLTP 'enabled' must be a boolean.")
        if not isinstance(self.volatility_window_bars, int) or self.volatility_window_bars <= 0:
            raise ValueError("volatility_window_bars must be a positive integer.")
        if not isinstance(self.fixed_take_profit_pct, (int, float)) or self.fixed_take_profit_pct <= 0:
            raise ValueError("fixed_take_profit_pct must be a positive number.")
        if not isinstance(self.fixed_stop_loss_pct, (int, float)) or self.fixed_stop_loss_pct <= 0:
            raise ValueError("fixed_stop_loss_pct must be a positive number.")
        if not isinstance(self.alpha_take_profit, (int, float)) or self.alpha_take_profit <= 0:
            raise ValueError("alpha_take_profit must be a positive number.")
        if not isinstance(self.alpha_stop_loss, (int, float)) or self.alpha_stop_loss <= 0:
            raise ValueError("alpha_stop_loss must be a positive number.")
        if not isinstance(self.min_sl_tp_pct, (int, float)) or self.min_sl_tp_pct < 0:
            raise ValueError("min_sl_tp_pct must be a non-negative number.")
        if self.min_sl_tp_pct > self.fixed_take_profit_pct or self.min_sl_tp_pct > self.fixed_stop_loss_pct:
            logger.warning("min_sl_tp_pct is greater than fixed_take_profit_pct or fixed_stop_loss_pct. This might lead to tighter-than-expected stops/targets if fixed values are used.")


@dataclass
class StrategyConfig:
    """
    Defines configuration parameters for the trading strategy itself,
    including risk management, entry/exit criteria, and trade sizing.
    """
    # --- Core Trade Management ---
    initial_capital: float = 5000.0         # Starting capital for the bot (both live and backtest).
    risk_per_trade_pct: float = 0.01        # Percentage of capital to risk per trade (e.g., 0.01 for 1%).
    leverage: int = 10                      # Leverage to use for futures trading.
    trading_fee_rate: float = 0.0005        # Maker/Taker fee rate (e.g., 0.0005 for 0.05%).

    # --- Entry Filters ---
    confidence_filter_enabled: bool = True  # Whether to use model probability thresholds for entry.
    confidence_threshold_long_pct: float = 0.65 # Min probability for a LONG signal to be considered.
    confidence_threshold_short_pct: float = 0.65 # Min probability for a SHORT signal to be considered.

    volatility_regime_filter_enabled: bool = False # Whether to filter trades based on volatility regimes.
    volatility_regime_params: VolatilityRegimeConfig = field(default_factory=VolatilityRegimeConfig) # Nested config for volatility regimes.

    trend_filter_enabled: bool = False      # Whether to use a simple EMA trend filter for entry.
    trend_filter_ema_period: int = 20       # EMA period for the trend filter. Price must be above EMA for long, below for short.

    allow_long_trades: bool = True          # Whether the strategy is allowed to open long positions.
    allow_short_trades: bool = True         # Whether the strategy is allowed to open short positions.

    # --- Exit Conditions ---
    exit_on_neutral_signal: bool = True     # Whether to close an open position if the model outputs a neutral (0) signal.
    slippage_tolerance_pct: float = 0.001   # Percentage to add/subtract from entry/exit price for simulated slippage (e.g., 0.001 for 0.1%).

    # SL/TP Configuration - now a nested dataclass
    sltp_params: SLTPConfig = field(default_factory=SLTPConfig)

    # Max Holding Period
    # If volatility_regime_filter_enabled is True, max holding is determined by regime_params.max_holding_bars
    # Otherwise, this default is used. Set to None for no max holding period by default.
    max_holding_period_bars_default: Optional[int] = 100 # Default max holding bars if no regime applies

    min_liq_distance_pct: float = 0.02      # Minimum percentage distance (e.g., 0.01 for 1%) to maintain
                                            # between SL and estimated liquidation price. If SL is closer, it's adjusted.

    # --- Live Trading Specifics (though some might apply to backtest setup) ---
    live_trade_update_interval_sec: int = 60 # Interval in seconds to check for new candles/update trade status.
    save_state_interval_minutes: int = 5    # How often to save the bot's state to disk during live trading.
    max_retries_api_call: int = 5           # Max retries for failed API calls.
    retry_delay_sec: float = 5.0            # Initial delay in seconds for API call retries (exponential backoff).
    ohlcv_data_buffer_size: int = 500       # Number of historical OHLCV bars to keep in memory for feature calculation.
    model_prediction_lookback_bars: int = 1 # How many bars back to get the prediction/probability for current decision.

    # --- NEW: For annualizing performance metrics ---
    bars_per_year: int = 105120 # Default for 5-minute bars (60/5 * 24 * 365)


    def __post_init__(self):
        """
        Performs validation for StrategyConfig parameters and initializes nested dataclasses.
        """
        if not isinstance(self.initial_capital, (int, float)) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be a positive number.")
        if not isinstance(self.risk_per_trade_pct, (int, float)) or not (0 < self.risk_per_trade_pct <= 1):
            raise ValueError("risk_per_trade_pct must be a float between 0 and 1 (exclusive of 0, inclusive of 1).")
        if not isinstance(self.leverage, int) or self.leverage <= 0:
            raise ValueError("leverage must be a positive integer.")
        if not isinstance(self.trading_fee_rate, (int, float)) or self.trading_fee_rate < 0:
            raise ValueError("trading_fee_rate must be a non-negative number.")

        if not isinstance(self.confidence_filter_enabled, bool):
            raise TypeError("confidence_filter_enabled must be a boolean.")
        if not isinstance(self.confidence_threshold_long_pct, (int, float)) or not (0 <= self.confidence_threshold_long_pct <= 1):
            raise ValueError("confidence_threshold_long_pct must be a float between 0 and 1.")
        if not isinstance(self.confidence_threshold_short_pct, (int, float)) or not (0 <= self.confidence_threshold_short_pct <= 1):
            raise ValueError("confidence_threshold_short_pct must be a float between 0 and 1.")
        
        if isinstance(self.volatility_regime_params, dict):
            self.volatility_regime_params = VolatilityRegimeConfig(**self.volatility_regime_params)
        elif not isinstance(self.volatility_regime_params, VolatilityRegimeConfig):
            raise TypeError("volatility_regime_params must be a dictionary or VolatilityRegimeConfig instance.")

        if not isinstance(self.trend_filter_enabled, bool):
            raise TypeError("trend_filter_enabled must be a boolean.")
        if not isinstance(self.trend_filter_ema_period, int) or self.trend_filter_ema_period <= 0:
            raise ValueError("trend_filter_ema_period must be a positive integer.")
        
        if not isinstance(self.allow_long_trades, bool):
            raise TypeError("allow_long_trades must be a boolean.")
        if not isinstance(self.allow_short_trades, bool):
            raise TypeError("allow_short_trades must be a boolean.")

        if not isinstance(self.exit_on_neutral_signal, bool):
            raise TypeError("exit_on_neutral_signal must be a boolean.")
        if not isinstance(self.slippage_tolerance_pct, (int, float)) or self.slippage_tolerance_pct < 0:
            raise ValueError("slippage_tolerance_pct must be a non-negative number.")
        
        if isinstance(self.sltp_params, dict):
            self.sltp_params = SLTPConfig(**self.sltp_params)
        elif not isinstance(self.sltp_params, SLTPConfig):
            raise TypeError("sltp_params must be a dictionary or SLTPConfig instance.")

        if self.max_holding_period_bars_default is not None and (not isinstance(self.max_holding_period_bars_default, int) or self.max_holding_period_bars_default <= 0):
            raise ValueError("max_holding_period_bars_default must be a positive integer or None.")

        if not isinstance(self.min_liq_distance_pct, (int, float)) or self.min_liq_distance_pct < 0:
            raise ValueError("min_liq_distance_pct must be a non-negative number.")
        
        if not isinstance(self.bars_per_year, int) or self.bars_per_year <= 0:
            raise ValueError("bars_per_year must be a positive integer.")

        # Live trading specific validation
        if not isinstance(self.live_trade_update_interval_sec, int) or self.live_trade_update_interval_sec <= 0:
            raise ValueError("live_trade_update_interval_sec must be a positive integer.")
        if not isinstance(self.save_state_interval_minutes, int) or self.save_state_interval_minutes <= 0:
            raise ValueError("save_state_interval_minutes must be a positive integer.")
        if not isinstance(self.max_retries_api_call, int) or self.max_retries_api_call < 0:
            raise ValueError("max_retries_api_call must be a non-negative integer.")
        if not isinstance(self.retry_delay_sec, (int, float)) or self.retry_delay_sec <= 0:
            raise ValueError("retry_delay_sec must be a positive number.")
        if not isinstance(self.ohlcv_data_buffer_size, int) or self.ohlcv_data_buffer_size <= 0:
            raise ValueError("ohlcv_data_buffer_size must be a positive integer.")
        if not isinstance(self.model_prediction_lookback_bars, int) or self.model_prediction_lookback_bars <= 0:
            raise ValueError("model_prediction_lookback_bars must be a positive integer.")


# Default configuration instance
DEFAULT_STRATEGY_CONFIG = StrategyConfig()
