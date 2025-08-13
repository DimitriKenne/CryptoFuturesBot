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
                raise TypeError(f"Value for volatility regime {k} in 'allow_trading' must be boolean. Got {type(v).__name__}: {v}")


@dataclass
class StrategyConfig:
    """
    Defines overall strategy-level configuration parameters, including risk management
    and various filters.
    """
    initial_capital: float = 10000.0 # Starting capital for backtesting and live trading
    risk_per_trade_pct: float = 0.01 # Max percentage of capital to risk per trade (e.g., 1%)
    leverage: float = 5.0 # Leverage to use for futures trading
    trading_fee_rate: float = 0.0005 # Per-trade fee rate (e.g., 0.05% for maker/taker)
    slippage_tolerance_pct: float = 0.0001 # Estimated slippage rate per transaction

    exit_on_neutral_signal: bool = True # Whether to close a position if the model outputs a 0 label
    allow_long_trades: bool = True # Set to False to disable long entries
    allow_short_trades: bool = True # Set to False to disable short entries

    # Confidence Threshold Filtering
    confidence_filter_enabled: bool = True
    confidence_threshold_long_pct: float = 0.65 # Minimum confidence (probability) for a long signal
    confidence_threshold_short_pct: float = 0.65 # Minimum confidence (probability) for a short signal

    # Volatility Regime Filtering
    volatility_regime_filter_enabled: bool = True
    volatility_regime_params: VolatilityRegimeConfig = field(default_factory=VolatilityRegimeConfig)

    # Trend Alignment Filter (EMA based)
    trend_filter_enabled: bool = True
    trend_filter_ema_period: int = 50 # Period for the EMA used in trend filtering

    sequence_length_bars: int = 5 # How many previous bars to use as input sequence for models (especially LSTM)
    analysis_future_horizons: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 150, 200]) # Horizons for label and backtest analysis
    live_trade_update_interval_sec: int = 30 # How often the live bot loop runs (in seconds)
    save_state_interval_minutes: int = 5 # How often to save bot state (capital, position)
    max_retries_api_call: int = 3 # Max retries for API calls
    retry_delay_sec: int = 5 # Delay between API call retries
    ohlcv_data_buffer_size: int = 200 # Number of recent OHLCV bars to keep in memory for live updates (should cover max indicator period)
    model_prediction_lookback_bars: int = 1 # How many bars back the model prediction applies (e.g., 1 if signal is for next bar)
    warm_up_period: int = 200 # Bars to wait before making first trade, allows indicators to stabilize

    def __post_init__(self):
        if not isinstance(self.initial_capital, (int, float)) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be a positive number.")
        if not isinstance(self.risk_per_trade_pct, float) or not (0 < self.risk_per_trade_pct < 1):
            raise ValueError("risk_per_trade_pct must be a float between 0 and 1 (exclusive).")
        if not isinstance(self.leverage, (int, float)) or self.leverage <= 0:
            raise ValueError("leverage must be a positive number.")
        if not isinstance(self.trading_fee_rate, float) or self.trading_fee_rate < 0:
            raise ValueError("trading_fee_rate must be a non-negative float.")
        if not isinstance(self.slippage_tolerance_pct, float) or self.slippage_tolerance_pct < 0:
            raise ValueError("slippage_tolerance_pct must be a non-negative float.")
        if not isinstance(self.exit_on_neutral_signal, bool):
            raise TypeError("exit_on_neutral_signal must be a boolean.")
        if not isinstance(self.allow_long_trades, bool):
            raise TypeError("allow_long_trades must be a boolean.")
        if not isinstance(self.allow_short_trades, bool):
            raise TypeError("allow_short_trades must be a boolean.")
        
        # Confidence thresholds
        if not isinstance(self.confidence_filter_enabled, bool):
            raise TypeError("confidence_filter_enabled must be a boolean.")
        if not isinstance(self.confidence_threshold_long_pct, float) or not (0 <= self.confidence_threshold_long_pct <= 1):
            raise ValueError("confidence_threshold_long_pct must be a float between 0 and 1.")
        if not isinstance(self.confidence_threshold_short_pct, float) or not (0 <= self.confidence_threshold_short_pct <= 1):
            raise ValueError("confidence_threshold_short_pct must be a float between 0 and 1.")
        
        # Volatility regime
        if not isinstance(self.volatility_regime_filter_enabled, bool):
            raise TypeError("volatility_regime_filter_enabled must be a boolean.")
        if isinstance(self.volatility_regime_params, dict):
            self.volatility_regime_params = VolatilityRegimeConfig(**self.volatility_regime_params)
        elif not isinstance(self.volatility_regime_params, VolatilityRegimeConfig):
             raise TypeError("volatility_regime_params must be a dictionary or VolatilityRegimeConfig instance.")

        # Trend filter
        if not isinstance(self.trend_filter_enabled, bool):
            raise TypeError("trend_filter_enabled must be a boolean.")
        if not isinstance(self.trend_filter_ema_period, int) or self.trend_filter_ema_period <= 0:
            raise ValueError("trend_filter_ema_period must be a positive integer.")
        
        # Other parameters
        if not isinstance(self.sequence_length_bars, int) or self.sequence_length_bars <= 0:
            raise ValueError("sequence_length_bars must be a positive integer.")
        if not isinstance(self.analysis_future_horizons, list) or not all(isinstance(x, int) and x > 0 for x in self.analysis_future_horizons):
            raise ValueError("analysis_future_horizons must be a list of positive integers.")
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
        if not isinstance(self.warm_up_period, int) or self.warm_up_period <= 0:
            raise ValueError("warm_up_period must be a positive integer.")


# Default configuration instance
DEFAULT_STRATEGY_CONFIG = StrategyConfig()
