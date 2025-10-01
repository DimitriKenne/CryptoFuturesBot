import streamlit as st
import json
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Ensure project root is in the path
# FIX: The `app.py` file is located inside `utils/bot_management`, so we need to
# go up two levels (`.parent.parent`) to get to the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import project-specific modules
from config.paths import PATH_CONFIG
from config.trading import DEFAULT_TRADING_CONFIG, TradingConfig
from utils.data_management.data_manager import DataManager

# --- Setup Logging ---
# Note: Streamlit has its own logging, so this is for explicit logs
# outside of the standard Streamlit output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def get_default_config() -> Dict[str, Any]:
    """Converts the default TradingConfig dataclass to a dictionary."""
    return {
        "symbol": DEFAULT_TRADING_CONFIG.symbol,
        "interval": DEFAULT_TRADING_CONFIG.interval,
        "model_type": DEFAULT_TRADING_CONFIG.model_type,
        "risk": DEFAULT_TRADING_CONFIG.risk.__dict__,
        "trade_execution": DEFAULT_TRADING_CONFIG.trade_execution.__dict__,
        "entry_filter": DEFAULT_TRADING_CONFIG.entry_filter.__dict__,
        "sltp": DEFAULT_TRADING_CONFIG.sltp.__dict__,
        "backtest": DEFAULT_TRADING_CONFIG.backtest.__dict__,
        "volatility_regime": DEFAULT_TRADING_CONFIG.volatility_regime.__dict__,
    }

# --- Streamlit UI ---
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("🛡️ Trading Bot Configuration")

# Sidebar for controls
st.sidebar.title("Configuration Options")
st.sidebar.markdown("Modify parameters and launch the bot.")

# Main configuration form
with st.form("config_form"):
    st.header("General Settings")
    symbol = st.text_input("Symbol", value=DEFAULT_TRADING_CONFIG.symbol)
    interval = st.selectbox("Interval", options=[
        '1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'], index=1)
    model_type = st.selectbox("Model Type", options=["random_forest", "xgboost", "lstm"], index=0)

    # Use tabs for a cleaner layout
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Risk", "Trade Execution", "Entry Filters", "SL/TP", "Backtest", "Volatility"
    ])

    # --- Risk Config Tab ---
    with tab1:
        st.subheader("Risk Management")
        risk_initial_capital = st.number_input("Initial Capital ($)", min_value=1.0, value=DEFAULT_TRADING_CONFIG.risk.initial_capital)
        risk_per_trade_pct = st.number_input("Risk Per Trade (%)", min_value=0.1, max_value=100.0, value=DEFAULT_TRADING_CONFIG.risk.risk_per_trade_pct)
        risk_leverage = st.number_input("Leverage", min_value=1, max_value=125, value=DEFAULT_TRADING_CONFIG.risk.leverage)

    # --- Trade Execution Tab ---
    with tab2:
        st.subheader("Trade Execution")
        trade_fee_pct = st.number_input("Trading Fee (%)", min_value=0.0, max_value=1.0, value=DEFAULT_TRADING_CONFIG.trade_execution.trading_fee_pct)
        slippage_tolerance_pct = st.number_input("Slippage Tolerance (%)", min_value=0.0, max_value=1.0, value=DEFAULT_TRADING_CONFIG.trade_execution.slippage_tolerance_pct)
        min_liq_distance_pct = st.number_input("Min Liquidation Distance (%)", min_value=0.0, max_value=100.0, value=DEFAULT_TRADING_CONFIG.trade_execution.min_liq_distance_pct)
        exit_on_neutral = st.checkbox("Exit on Neutral Signal", value=DEFAULT_TRADING_CONFIG.trade_execution.exit_on_neutral_signal)

    # --- Entry Filter Tab ---
    with tab3:
        st.subheader("Entry Filters")
        conf_filter_enabled = st.checkbox("Enable Confidence Filter", value=DEFAULT_TRADING_CONFIG.entry_filter.confidence_filter_enabled)
        conf_threshold_long = st.number_input("Confidence Threshold Long (%)", min_value=0, max_value=100, value=DEFAULT_TRADING_CONFIG.entry_filter.confidence_threshold_long_pct)
        conf_threshold_short = st.number_input("Confidence Threshold Short (%)", min_value=0, max_value=100, value=DEFAULT_TRADING_CONFIG.entry_filter.confidence_threshold_short_pct)
        trend_filter_enabled = st.checkbox("Enable Trend Filter", value=DEFAULT_TRADING_CONFIG.entry_filter.trend_filter_enabled)
        trend_filter_ema_period = st.number_input("Trend Filter EMA Period", min_value=1, value=DEFAULT_TRADING_CONFIG.entry_filter.trend_filter_ema_period)
        allow_long = st.checkbox("Allow Long Trades", value=DEFAULT_TRADING_CONFIG.entry_filter.allow_long_trades)
        allow_short = st.checkbox("Allow Short Trades", value=DEFAULT_TRADING_CONFIG.entry_filter.allow_short_trades)

    # --- SL/TP Config Tab ---
    with tab4:
        st.subheader("Stop Loss / Take Profit")
        sltp_enabled = st.checkbox("Enable Dynamic SL/TP", value=DEFAULT_TRADING_CONFIG.sltp.enabled)
        volatility_window = st.number_input("Volatility Window (bars)", min_value=1, value=DEFAULT_TRADING_CONFIG.sltp.volatility_window_bars)
        fixed_tp_pct = st.number_input("Fixed Take Profit (%)", min_value=0.1, value=DEFAULT_TRADING_CONFIG.sltp.fixed_take_profit_pct)
        fixed_sl_pct = st.number_input("Fixed Stop Loss (%)", min_value=0.1, value=DEFAULT_TRADING_CONFIG.sltp.fixed_stop_loss_pct)
        alpha_tp = st.number_input("ATR Take Profit Multiplier", min_value=0.1, value=DEFAULT_TRADING_CONFIG.sltp.alpha_take_profit)
        alpha_sl = st.number_input("ATR Stop Loss Multiplier", min_value=0.1, value=DEFAULT_TRADING_CONFIG.sltp.alpha_stop_loss)
        min_sl_tp_pct = st.number_input("Min SL/TP Distance (%)", min_value=0.1, value=DEFAULT_TRADING_CONFIG.sltp.min_sl_tp_pct)
        max_holding_period = st.number_input("Max Holding Period (bars)", min_value=1, value=DEFAULT_TRADING_CONFIG.sltp.max_holding_period_bars_default)

    # --- Backtest Config Tab ---
    with tab5:
        st.subheader("Backtesting")
        maintenance_margin_pct = st.number_input("Maintenance Margin (%)", min_value=0.0, max_value=100.0, value=DEFAULT_TRADING_CONFIG.backtest.maintenance_margin_pct)
        liquidation_fee_pct = st.number_input("Liquidation Fee (%)", min_value=0.0, max_value=100.0, value=DEFAULT_TRADING_CONFIG.backtest.liquidation_fee_pct)
        max_concurrent_trades = st.number_input("Max Concurrent Trades", min_value=1, value=DEFAULT_TRADING_CONFIG.backtest.max_concurrent_trades)
        backtest_mode = st.selectbox("Backtest Mode", options=["full", "train", "test"], index=2)
        save_trades = st.checkbox("Save Trades", value=DEFAULT_TRADING_CONFIG.backtest.save_trades)
        save_equity_curve = st.checkbox("Save Equity Curve", value=DEFAULT_TRADING_CONFIG.backtest.save_equity_curve)
        save_metrics = st.checkbox("Save Metrics", value=DEFAULT_TRADING_CONFIG.backtest.save_metrics)

    # --- Volatility Regime Tab ---
    with tab6:
        st.subheader("Volatility Regime")
        st.markdown("Configure max holding bars and trading permissions per volatility regime (0, 1, 2).")
        vr_max_holding_0 = st.number_input("Max Holding Bars (Regime 0)", min_value=1, value=DEFAULT_TRADING_CONFIG.volatility_regime.max_holding_bars.get(0, 300))
        vr_allow_trading_0 = st.checkbox("Allow Trading (Regime 0)", value=DEFAULT_TRADING_CONFIG.volatility_regime.allow_trading.get(0, True))
        vr_max_holding_1 = st.number_input("Max Holding Bars (Regime 1)", min_value=1, value=DEFAULT_TRADING_CONFIG.volatility_regime.max_holding_bars.get(1, 200))
        vr_allow_trading_1 = st.checkbox("Allow Trading (Regime 1)", value=DEFAULT_TRADING_CONFIG.volatility_regime.allow_trading.get(1, True))
        vr_max_holding_2 = st.number_input("Max Holding Bars (Regime 2)", min_value=1, value=DEFAULT_TRADING_CONFIG.volatility_regime.max_holding_bars.get(2, 150))
        vr_allow_trading_2 = st.checkbox("Allow Trading (Regime 2)", value=DEFAULT_TRADING_CONFIG.volatility_regime.allow_trading.get(2, True))

    # Form submission button
    submitted = st.form_submit_button("Launch Bot")

# --- Form Submission Logic ---
if submitted:
    try:
        # Construct the user's config dictionary
        user_config = {
            "symbol": symbol,
            "interval": interval,
            "model_type": model_type,
            "risk": {
                "initial_capital": risk_initial_capital,
                "risk_per_trade_pct": risk_per_trade_pct,
                "leverage": risk_leverage,
            },
            "trade_execution": {
                "trading_fee_pct": trade_fee_pct,
                "slippage_tolerance_pct": slippage_tolerance_pct,
                "min_liq_distance_pct": min_liq_distance_pct,
                "exit_on_neutral_signal": exit_on_neutral,
            },
            "entry_filter": {
                "confidence_filter_enabled": conf_filter_enabled,
                "confidence_threshold_long_pct": conf_threshold_long,
                "confidence_threshold_short_pct": conf_threshold_short,
                "trend_filter_enabled": trend_filter_enabled,
                "trend_filter_ema_period": trend_filter_ema_period,
                "allow_long_trades": allow_long,
                "allow_short_trades": allow_short,
            },
            "sltp": {
                "enabled": sltp_enabled,
                "volatility_window_bars": volatility_window,
                "fixed_take_profit_pct": fixed_tp_pct,
                "fixed_stop_loss_pct": fixed_sl_pct,
                "alpha_take_profit": alpha_tp,
                "alpha_stop_loss": alpha_sl,
                "min_sl_tp_pct": min_sl_tp_pct,
                "max_holding_period_bars_default": max_holding_period,
            },
            "backtest": {
                "maintenance_margin_pct": maintenance_margin_pct,
                "liquidation_fee_pct": liquidation_fee_pct,
                "max_concurrent_trades": max_concurrent_trades,
                "backtest_mode": backtest_mode,
                "save_trades": save_trades,
                "save_equity_curve": save_equity_curve,
                "save_metrics": save_metrics,
            },
            "volatility_regime": {
                "max_holding_bars": {
                    0: vr_max_holding_0,
                    1: vr_max_holding_1,
                    2: vr_max_holding_2,
                },
                "allow_trading": {
                    0: vr_allow_trading_0,
                    1: vr_allow_trading_1,
                    2: vr_allow_trading_2,
                }
            },
        }

        # Validate and instantiate the TradingConfig dataclass
        current_app_config = TradingConfig.from_dict(user_config)

        # We'll save the config as a JSON file to be read by the trading bot
        # This requires the trading bot to be able to read a config file, which
        # it seems to do based on the snippet you provided.
        
        # Save the configuration file using DataManager
        dm = DataManager(path_config=PATH_CONFIG)
        try:
            config_file_path = dm.save_app_config(
                config_data=user_config,
                model_type=model_type,
                symbol=symbol,
                interval=interval
            )
            
            # Launch the trading bot as a subprocess
            st.success(f"Configuration saved to {config_file_path}. Launching bot...")
            
            # Command to run the trading bot
            command = [
                sys.executable,  # Use the same Python interpreter
                str(PROJECT_ROOT / "trading_bot.py"),
                "--config_file", str(config_file_path),
                "--mode", "live", # Assuming live mode from the UI
            ]
            
            # Use st.spinner for a loading indicator
            with st.spinner('Bot is launching... Check your console for logs.'):
                # We'll run this in a non-blocking way so Streamlit doesn't freeze
                subprocess.Popen(command)
            
            st.info("Bot launched! See your local console for real-time logs.")

        except Exception as e:
            st.error(f"Failed to launch the bot. Error: {e}")
            logger.error(f"Error during bot launch: {e}", exc_info=True)

    except Exception as e:
        st.error(f"Failed to create a valid configuration from inputs. Error: {e}")
        logger.error(f"Error during config creation: {e}", exc_info=True)
