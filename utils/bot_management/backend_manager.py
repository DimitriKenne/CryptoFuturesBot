import asyncio
import logging
import signal
import sys
import os
import uuid
from pathlib import Path
from typing import Dict, Any
import json
import sqlite3
import subprocess

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Define the project root and add it to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import your project's modules
from utils.data_management.data_manager import DataManager
from config.paths import PATH_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory dictionary to track running bot instances
RUNNING_BOTS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Trading Bot Process Manager")

class BotConfig(BaseModel):
    """Pydantic model for the incoming bot configuration."""
    model_type: str
    symbol: str
    interval: str
    bot_mode: str # NEW: Add bot_mode
    config_data: Dict[str, Any]

@app.post("/bots/start")
async def start_bot_instance(bot_config: BotConfig):
    """Starts a new trading bot instance as a background process."""
    try:
        # Generate a unique bot ID based on the configuration
        bot_id = f"{bot_config.model_type}-{bot_config.symbol}-{bot_config.interval}"
        if bot_id in RUNNING_BOTS:
            raise HTTPException(status_code=400, detail=f"Bot with ID '{bot_id}' is already running.")

        # Corrected: Initialize DataManager without arguments
        dm = DataManager()
        
        # Save the configuration file using the existing DataManager
        config_file_path = dm.save_app_config(
            config_data=bot_config.config_data,
            model_type=bot_config.model_type,
            symbol=bot_config.symbol,
            interval=bot_config.interval,
        )

        # Build the command to run the trading bot
        command = [
            sys.executable,
            str(PROJECT_ROOT / "trading_bot.py"),
            "--config_file", str(config_file_path),
            "--mode", bot_config.bot_mode, # UPDATED: Use the mode from the request,
            # --- CORRECTED: Added the missing arguments ---
            "--symbol", bot_config.symbol,
            "--interval", bot_config.interval,
            "--model_type", bot_config.model_type,
            # ---------------------------------------------
        ]

        # Corrected: Removed preexec_fn=os.setsid for Windows compatibility
        process = subprocess.Popen(command, text=True, bufsize=1, universal_newlines=True)

        # Store process information in our in-memory dictionary
        run_dir = dm.get_live_trading_dir(bot_config.model_type, bot_config.symbol, bot_config.interval)
        RUNNING_BOTS[bot_id] = {
            "process": process,
            "config_path": str(config_file_path),
            "pid": process.pid,
            "run_dir": str(run_dir),
            "status": "running",
            "model_type": bot_config.model_type,
            "symbol": bot_config.symbol,
            "interval": bot_config.interval,
        }

        logger.info(f"Bot instance '{bot_id}' started with PID: {process.pid}")
        return {"bot_id": bot_id, "status": "launched"}

    except Exception as e:
        logger.error(f"Failed to start bot instance: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start bot instance: {e}")

@app.get("/bots/status")
async def get_all_bot_statuses():
    """Returns the current status of all running bot instances."""
    statuses = []
    
    # Use a copy of the dict to avoid issues if a bot stops mid-loop
    bots_to_check = list(RUNNING_BOTS.keys())
    for bot_id in bots_to_check:
        bot_info = RUNNING_BOTS.get(bot_id)
        if not bot_info:
            continue

        process = bot_info["process"]
        # Check if the process is still running
        if process.poll() is not None:
            # The process has terminated, clean it up
            logger.warning(f"Bot '{bot_id}' with PID {process.pid} has stopped unexpectedly. Cleaning up.")
            del RUNNING_BOTS[bot_id]
            continue
        
        # Read the latest state from the bot_state.db file
        bot_state = {}
        try:
            # Note: We'll use the DataManager to get the path and load the state
            dm = DataManager()
            bot_state = dm.load_bot_state(
                model_type=bot_info['model_type'],
                symbol=bot_info['symbol'],
                interval=bot_info['interval']
            )
        except Exception as e:
            logger.error(f"Could not read state for bot '{bot_id}': {e}")
            bot_state = {"error": "Could not read state"}
        
         # --- START MODIFICATION ---
        # Extract position and PnL information
        current_position = bot_state.get("current_position", {}) # Use an empty dict if key is missing
        
        statuses.append({
            "bot_id": bot_id,
            "pid": process.pid,
            "status": "running",
            "last_update": bot_state.get("last_processed_timestamp"),
            "current_capital": bot_state.get("current_capital"),
            "model_type": bot_info["model_type"],
            "symbol": bot_info["symbol"],
            "interval": bot_info["interval"],
            # ADDED FIELDS FOR FRONTEND DISPLAY
            "position_size": current_position.get("size", 0),
            "unrealized_pnl": current_position.get("unrealized_pnl", 0.0), 
        })
        # --- END MODIFICATION ---
    
    return statuses

@app.post("/bots/shutdown/{bot_id}")
async def shutdown_bot_instance(bot_id: str):
    """Sends a shutdown signal to a specific bot instance."""
    bot_info = RUNNING_BOTS.get(bot_id)
    if not bot_info:
        raise HTTPException(status_code=404, detail="Bot not found.")

    process = bot_info["process"]
    try:
        # Send a graceful shutdown signal (SIGTERM)
        process.terminate()
        logger.info(f"Shutdown signal sent to bot '{bot_id}' (PID: {process.pid}).")
        
        # Clean up the entry from our dictionary immediately
        # The next status check will confirm the process is gone
        del RUNNING_BOTS[bot_id]

        return {"message": f"Shutdown initiated for bot '{bot_id}'."}
    except Exception as e:
        logger.error(f"Failed to shutdown bot '{bot_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to shutdown bot: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# You must run the following command to start the server:
# uvicorn utils.bot_management.backend_manager:app --host 0.0.0.0 --port 8000
# NOTE: Do NOT use --reload if you want running bots to persist config changes.