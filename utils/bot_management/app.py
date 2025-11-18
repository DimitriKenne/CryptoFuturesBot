import streamlit as st
import json
import requests
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from dataclasses import is_dataclass, asdict

# Ensure project root is in the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Import project-specific modules
from config.paths import PATH_CONFIG
from config.trading import DEFAULT_TRADING_CONFIG
from utils.data_management.data_manager import DataManager

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
BACKEND_URL = "http://localhost:8000"

# --- Helper Functions ---
def get_default_config() -> Dict[str, Any]:
    """Dynamically converts the default TradingConfig dataclass to a dictionary."""
    return asdict(DEFAULT_TRADING_CONFIG)

def start_bot(config_data: Dict[str, Any], symbol: str, interval: str, model_type: str, bot_mode: str="automatic"):
    """Sends a request to the backend service to start a new bot instance."""
    payload = {
        "model_type": model_type,
        "symbol": symbol,
        "interval": interval,
        "bot_mode": bot_mode, # ADDED: Pass bot_mode to the backend
        "config_data": config_data,
    }
    try:
        response = requests.post(f"{BACKEND_URL}/bots/start", json=payload)
        response.raise_for_status()
        st.success(f"Bot launched successfully! Bot ID: {response.json().get('bot_id')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to the backend service. Error: {e}")

def shutdown_bot(bot_id: str):
    """Sends a request to the backend service to shut down a specific bot instance."""
    try:
        response = requests.post(f"{BACKEND_URL}/bots/shutdown/{bot_id}")
        response.raise_for_status()
        st.info(f"Shutdown signal sent for bot ID: {bot_id}.")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to send shutdown signal. Error: {e}")

# --- Streamlit UI ---
st.set_page_config(
    page_title="Trading Bot Management Dashboard",
    layout="wide",
)

st.title("🤖 Trading Bot Management Dashboard")

# Create two columns for the form and the dashboard
form_col, dashboard_col = st.columns([1, 2])

with form_col:
    st.header("Launch a New Bot Instance")
    st.info("Fill out the configuration to launch a new bot.")

    # A simple form to get the core config details from the user
    user_symbol = st.selectbox(
        "Trading Symbol",
        options=["FLOWUSDT", "1000SHIBUSDT", "ADAUSDT", "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"],
        index=0,
    )
    user_interval = st.selectbox(
        "Time Interval",
        options=["1m", "5m", "15m", "1h", "4h"],
        index=1,
    )
    user_model_type = st.selectbox(
        "Model Type",
        options=["random_forest", "xgboost", "lstm"],
        index=0,
    )
    # NEW: Bot Mode selection
    user_bot_mode = st.selectbox(
        "Bot Mode",
        options=["automatic", "hybrid"],
        index=0,
        help="Select 'automatic' for fully autonomous trading or 'hybrid' for a mode requiring user/external approval for trades."
    )

    # Use a text area for the full JSON configuration
    st.subheader("Configuration Override (Optional)")
    default_config = get_default_config()
    default_config_str = json.dumps(default_config, indent=4)
    user_config_input = st.text_area(
        "Modify the configuration JSON here:",
        value=default_config_str,
        height=400,
    )

    if st.button("🚀 Launch Bot"):
        try:
            # Parse the user's JSON input
            user_config = json.loads(user_config_input)

            # Update the user's config with the selected symbol/interval
            user_config['symbol'] = user_symbol
            user_config['interval'] = user_interval
            user_config['model_type'] = user_model_type
            
            # Call the start_bot function
            start_bot(user_config, user_symbol, user_interval, user_model_type)

        except json.JSONDecodeError:
            st.error("Invalid JSON format. Please check your configuration input.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

with dashboard_col:
    st.header("Active Bots Dashboard")
    st.info("This table shows all bots managed by the backend service.")
    
    # Use st.empty to create a placeholder that we will update
    status_placeholder = st.empty()

    def update_dashboard():
        """Fetches and updates the bot statuses."""
        try:
            response = requests.get(f"{BACKEND_URL}/bots/status")
            response.raise_for_status()
            bots_status = response.json()
            
            if not bots_status:
                status_placeholder.write("No active bots found.")
                return

            df = pd.DataFrame(bots_status)
            
            # MODIFICATION 1: Update the number of columns (6 + 2 new columns = 8)
            cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
            
            # MODIFICATION 2: Update the column names
            col_names = [
                "Bot ID", "Symbol", "Status", 
                "Capital", "Pos. Size", "Unrealized PnL", 
                "Last Update", "Action"
            ]
            for col, col_name in zip(cols, col_names):
                col.write(f"**{col_name}**")

            for index, row in df.iterrows():
                with st.container():
                    # MODIFICATION 3: Use 8 columns to match the header
                    cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
                    
                    # Format PnL for display, handling None/NaN
                    unrealized_pnl = row.get('unrealized_pnl')
                    if pd.isna(unrealized_pnl) or unrealized_pnl is None:
                        pnl_display = "N/A"
                        pnl_color = 'black'
                    else:
                        pnl_display = f"{unrealized_pnl:+.2f}" # Added '+' for positive PnL
                        pnl_color = 'green' if unrealized_pnl > 0 else 'red' if unrealized_pnl < 0 else 'gray'


                    # Col 0: Bot ID
                    cols[0].write(row['bot_id'])
                    
                    # Col 1: Symbol/Interval (Combined for space)
                    cols[1].write(f"{row['symbol']}/{row['interval']}")

                    # Col 2: Status
                    cols[2].write(row['status'])

                    # Col 3: Capital
                    cols[3].write(f"{row['current_capital']:.2f}" if row['current_capital'] is not None else "N/A")

                    # Col 4: Position Size
                    pos_size = row.get('position_size', 0)
                    cols[4].write(pos_size)
                    
                    # Col 5: Unrealized PnL (Applying color logic)
                    cols[5].markdown(f":{pnl_color}[{pnl_display}]")

                    # Col 6: Last Update
                    cols[6].write(row['last_update'] if row['last_update'] is not None else "N/A")
                    
                    # Col 7: Action Button
                    if cols[7].button("Shutdown", key=f"shutdown_{row['bot_id']}"):
                        shutdown_bot(row['bot_id'])
                        time.sleep(1)
                        st.rerun()
            
        except requests.exceptions.RequestException as e:
            status_placeholder.error(f"Failed to connect to backend: {e}")
            
    update_dashboard()
    
    # MODIFICATION 4: Use st.rerun() instead of time.sleep(5) and st.rerun() together, 
    # and rely on the Streamlit widget-based rerun mechanism for automatic updates.
    # However, since you were using a 5-second sleep for a simple auto-refresh loop,
    # I will stick to your pattern but use st.rerun() alone for cleaner logic.
    time.sleep(5)
    st.rerun()

# How to run:
# 1. Start the backend service (utils/bot_management/app.py).               
# 2. Run this Streamlit app using: streamlit run utils/bot_management/app.py