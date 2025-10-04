from dataclasses import dataclass, field, is_dataclass
from typing import Dict, Any, Optional, Literal

# --- Sub-configs ---
@dataclass
class RiskConfig:
    initial_capital: float = 115.0
    risk_per_trade_pct: float = 5.0 # Percentage of capital to risk per trade
    leverage: int = 15

@dataclass
class TradeExecutionConfig:
    trading_fee_pct: float = 0.05 # Trading fee as a percentage
    slippage_tolerance_pct: float = 0.01 # Slippage tolerance as a percentage
    min_liq_distance_pct: float = 1.0 # Minimum distance from liquidation price as a percentage
    exit_on_neutral_signal: bool = False # Whether to exit on a neutral signal

@dataclass
class EntryFilterConfig:
    confidence_filter_enabled: bool = True
    confidence_threshold_long_pct: float = 60 # Confidence threshold for long entries (percentage)
    confidence_threshold_short_pct: float = 60 # Confidence threshold for short entries (percentage)
    volatility_regime_filter_enabled: bool = False
    trend_filter_enabled: bool = False
    trend_filter_ema_period: int = 50
    allow_long_trades: bool = True
    allow_short_trades: bool = True

@dataclass
class VolatilityRegimeConfig:
    max_holding_bars: Dict[int, int] = field(default_factory=lambda: {0: 300, 1: 200, 2: 150}) # Max holding bars per volatility regime
    allow_trading: Dict[int, bool] = field(default_factory=lambda: {0: True, 1: True, 2: True}) # Allow trading per volatility regime

@dataclass
class SLTPConfig:
    enabled: bool = False # Enable dynamic SLTP strategy
    volatility_window_bars: int = 20 # Window for ATR calculation
    fixed_take_profit_pct: float = 10.0 # Fixed Take Profit as a percentage
    fixed_stop_loss_pct: float = 4.0 # Fixed Stop Loss as a percentage
    alpha_take_profit: float = 14.0 # Multiplier for ATR-based TP
    alpha_stop_loss: float = 5.0 # Multiplier for ATR-based SL
    min_sl_tp_pct: float = 1.0 # Minimum SL/TP distance as a percentage
    max_holding_period_bars_default: Optional[int] = 300 # Default max holding if no regime applies


@dataclass
class BacktestConfig:
    maintenance_margin_pct: float = 0.5 # Maintenance margin as a percentage
    liquidation_fee_pct: float = 0.05 # Liquidation fee as a percentage
    max_concurrent_trades: int = 1
    backtest_mode: Literal['full', 'train', 'test'] = 'test'
    save_trades: bool = True
    save_equity_curve: bool = True
    save_metrics: bool = True
    monte_carlo_iterations: int = 10
    monte_carlo_seed: Optional[int] = None
    monte_carlo_plot_simulations: int = 20 # Number of MC simulations to plot
    override_strategy_params: Dict[str, Any] = field(default_factory=dict)

# --- Aggregated Top-Level Config ---
@dataclass
class TradingConfig:
    """
    Aggregated trading strategy and backtesting configurations.
    """
    
    # Symbol and model settings
    symbol: str = "1000SHIBUSDT"
    interval: str = "5m"
    model_type: Literal['random_forest', 'xgboost', 'lstm'] = 'random_forest'

    # Risk management settings
    risk: RiskConfig = field(default_factory=RiskConfig)
    trade_execution: TradeExecutionConfig = field(default_factory=TradeExecutionConfig)
    entry_filter: EntryFilterConfig = field(default_factory=EntryFilterConfig)
    volatility_regime: VolatilityRegimeConfig = field(default_factory=VolatilityRegimeConfig)
    sltp: SLTPConfig = field(default_factory=SLTPConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    bars_per_year: int = 105120 # Number of bars in a year for annualization
    
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """
        Recursively creates a TradingConfig instance from a dictionary.
        
        Args:
            data: A dictionary containing new configuration values.
            
        Returns:
            A new, frozen TradingConfig instance.
        """
        kwargs = {}
        for field_name, field_type in cls.__annotations__.items():
            actual_type = field_type.__args__[0] if getattr(field_type, '__origin__', None) is field else field_type
            
            if is_dataclass(actual_type):
                if isinstance(data.get(field_name), dict):
                    nested_data = data.get(field_name).copy()

                    # Special handling for VolatilityRegimeConfig
                    if actual_type.__name__ == 'VolatilityRegimeConfig':
                        for key in ['max_holding_bars', 'allow_trading']:
                            if key in nested_data and isinstance(nested_data[key], dict):
                                try:
                                    # Convert string keys to integers
                                    nested_data[key] = {int(k): v for k, v in nested_data[key].items()}
                                except (ValueError, TypeError) as e:
                                    raise TypeError(f"Invalid key type in `{key}`. Keys must be convertible to integers. Error: {e}")

                    # Use a specific from_dict method if it exists, otherwise instantiate
                    if 'from_dict' in dir(actual_type):
                        kwargs[field_name] = actual_type.from_dict(nested_data)
                    else:
                        kwargs[field_name] = actual_type(**nested_data)
                else:
                    kwargs[field_name] = field_type.default_factory() if 'default_factory' in str(field_type) else actual_type()
            else:
                if field_name in data:
                    kwargs[field_name] = data[field_name]
                else:
                    kwargs[field_name] = getattr(cls(), field_name)
        
        return cls(**kwargs)

# Default config instance
DEFAULT_TRADING_CONFIG = TradingConfig()
