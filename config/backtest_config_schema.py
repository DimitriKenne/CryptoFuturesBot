# config/backtest_config_schema.py

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class BacktestConfig:
    """
    Defines configuration parameters specifically for the backtesting engine.
    """
    # Core Backtester Mechanics
    maintenance_margin_rate: float = 0.005  # Estimated maintenance margin rate for liquidation calculation (e.g., 0.005 = 0.5%).
    liquidation_fee_rate: float = 0.0005    # Specific fee rate applied on simulated liquidation (defaults to trading_fee_rate if omitted, but explicit here).
    max_concurrent_trades: int = 1          # Maximum simultaneous open trades allowed during backtest. (Currently only 1 is fully supported for simplicity).

    # Reporting Flags
    save_trades: bool = True                # Whether to save detailed trade logs to file.
    save_equity_curve: bool = True          # Whether to save equity curve data to file.
    save_metrics: bool = True               # Whether to save summary performance metrics to file.

    # Strategy Overrides for Backtesting (Optional)
    # This dictionary allows users to override specific parameters from StrategyConfig
    # solely for backtesting purposes without changing the live trading configuration.
    # The keys should match attribute names in StrategyConfig, and values are the overrides.
    override_strategy_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Performs validation for BacktestConfig parameters.
        """
        if not isinstance(self.maintenance_margin_rate, (int, float)) or self.maintenance_margin_rate < 0:
            raise ValueError("maintenance_margin_rate must be a non-negative number.")
        if not isinstance(self.liquidation_fee_rate, (int, float)) or self.liquidation_fee_rate < 0:
            raise ValueError("liquidation_fee_rate must be a non-negative number.")
        if not isinstance(self.max_concurrent_trades, int) or self.max_concurrent_trades <= 0:
            raise ValueError("max_concurrent_trades must be a positive integer.")
        if not isinstance(self.save_trades, bool):
            raise TypeError("save_trades must be a boolean.")
        if not isinstance(self.save_equity_curve, bool):
            raise TypeError("save_equity_curve must be a boolean.")
        if not isinstance(self.save_metrics, bool):
            raise TypeError("save_metrics must be a boolean.")
        if not isinstance(self.override_strategy_params, dict):
            raise TypeError("override_strategy_params must be a dictionary.")

# Default configuration instance
DEFAULT_BACKTEST_CONFIG = BacktestConfig()
