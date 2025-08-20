from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal

# --- Sub-configs ---
@dataclass
class RiskConfig:
    initial_capital: float = 5000.0
    risk_per_trade_pct: float = 1.0 # Percentage of capital to risk per trade
    leverage: int = 10

@dataclass
class TradeExecutionConfig:
    trading_fee_pct: float = 0.05 # Trading fee as a percentage
    slippage_tolerance_pct: float = 0.05 # Slippage tolerance as a percentage
    min_liq_distance_pct: float = 2.0 # Minimum distance from liquidation price as a percentage
    exit_on_neutral_signal: bool = False # Whether to exit on a neutral signal

@dataclass
class EntryFilterConfig:
    confidence_filter_enabled: bool = True
    confidence_threshold_long_pct: float = 10 # Confidence threshold for long entries (percentage)
    confidence_threshold_short_pct: float = 10 # Confidence threshold for short entries (percentage)
    volatility_regime_filter_enabled: bool = False
    trend_filter_enabled: bool = False
    trend_filter_ema_period: int = 20
    allow_long_trades: bool = True
    allow_short_trades: bool = True

@dataclass
class VolatilityRegimeConfig:
    max_holding_bars: Dict[int, int] = field(default_factory=lambda: {0: 200, 1: 100, 2: 50}) # Max holding bars per volatility regime
    allow_trading: Dict[int, bool] = field(default_factory=lambda: {0: True, 1: True, 2: True}) # Allow trading per volatility regime

@dataclass
class SLTPConfig:
    enabled: bool = True # Enable dynamic SLTP strategy
    volatility_window_bars: int = 20 # Window for ATR calculation
    fixed_take_profit_pct: float = 6.0 # Fixed Take Profit as a percentage
    fixed_stop_loss_pct: float = 2.0 # Fixed Stop Loss as a percentage
    alpha_take_profit: float = 14.0 # Multiplier for ATR-based TP
    alpha_stop_loss: float = 5.0 # Multiplier for ATR-based SL
    min_sl_tp_pct: float = 1.0 # Minimum SL/TP distance as a percentage
    max_holding_period_bars_default: Optional[int] = None # Default max holding if no regime applies


@dataclass
class BacktestConfig:
    maintenance_margin_pct: float = 0.5 # Maintenance margin as a percentage
    liquidation_fee_pct: float = 0.05 # Liquidation fee as a percentage
    max_concurrent_trades: int = 1
    backtest_mode: Literal['full', 'train', 'test'] = 'test'
    save_trades: bool = True
    save_equity_curve: bool = True
    save_metrics: bool = True
    monte_carlo_iterations: int = 1000
    monte_carlo_seed: Optional[int] = None
    monte_carlo_plot_simulations: int = 20 # Number of MC simulations to plot
    override_strategy_params: Dict[str, Any] = field(default_factory=dict)

# --- Aggregated Top-Level Config ---
@dataclass
class TradingConfig:
    """
    Aggregated trading strategy and backtesting configurations.
    """
    risk: RiskConfig = field(default_factory=RiskConfig)
    trade_execution: TradeExecutionConfig = field(default_factory=TradeExecutionConfig)
    entry_filter: EntryFilterConfig = field(default_factory=EntryFilterConfig)
    volatility_regime: VolatilityRegimeConfig = field(default_factory=VolatilityRegimeConfig)
    sltp: SLTPConfig = field(default_factory=SLTPConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    bars_per_year: int = 105120 # Number of bars in a year for annualization

# Default config instance
DEFAULT_TRADING_CONFIG = TradingConfig()
