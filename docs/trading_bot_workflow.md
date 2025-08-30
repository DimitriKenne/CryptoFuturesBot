# Trading Bot - Complete Operational Workflow

**Author:** DimitriKenne  
**Date:** 2025-08-30 06:00:00 UTC (updated)

This document outlines the definitive, end-to-end operational workflow for the live trading bot, including support for "Hybrid mode" with a robust CLI menu, improved input validation, timeout handling, and **robust SL/TP protection reconstruction** on startup.

---

## Phase 1: Initialization and Startup (Robust Reconciliation)

This phase ensures the bot starts in a clean, known, and synchronized state using a comprehensive, two-phase reconciliation process.

1. **Load Configuration:** The bot starts and loads all necessary parameters from the `AppConfig` object, including the unique `bot_id` and the selected `mode` (`automatic` or `hybrid`).
2. **Initialize Core Components:** It creates instances of all core modules (`LifecycleManager`, `TradeCycleProcessor`, etc.).
3. **Send Startup Notification:** "Bot is starting up..."
4. **Connect to Exchange:** The `BinanceFuturesAdapter` connects to the API and fetches initial exchange rules.
5. **Load Persistent State:** The bot loads the last known state (`current_capital`, `open_position`, etc.) from the `state.json` file into the `LiveTradingSessionManager`.

6. **Critical - The Reconciliation Loop:** This is the core of a safe startup.

    *   **Phase A: Fetch Ground Truth**
        1.  Fetch ALL open positions for the trading symbol from the exchange.
        2.  Fetch ALL open orders for the trading symbol from the exchange.

    *   **Phase B: Reconcile State & Enforce Protection**
        1.  **Reconcile Active Position:**
            *   **Match Found:** If the exchange has one position matching the bot's loaded position (by order ID or price/qty), it is adopted.
            *   **Orphan Found:** If the exchange has a position but the bot has no state, the position is adopted into bot state.
            *   **Ghost Found:** If the bot has a position in its state but the exchange does not, a `WARNING` is logged, and the position is cleared from the bot's state.
            *   **Multiple Positions:** If the exchange shows multiple positions, a `WARNING` is logged, and an attempt is made to close the smaller, unexpected positions to consolidate into one.

        2.  **Reconcile SL/TP Orders:**
            *   Based on the now-synced active position, the bot knows what its ideal SL/TP orders should be (by order ID).
            *   **Missing Protection:** If an ideal SL or TP order is *missing* from the exchange, the bot IMMEDIATELY and AUTOMATICALLY **replaces all missing protection orders** using `TradeCycleProcessor.ensure_sltp_orders()`. The state is then saved.
            *   **Orphan Orders:** Any open order for the symbol on the exchange that is *not* the reconciled position's verified SL or TP order (by order ID) is considered an orphan and is cancelled.

7. **Launch Main Loop:** Only after the state is fully reconciled and the position (if any) is confirmed to be protected by SL/TP orders does the main trading loop begin.
8. **Send Operational Notification:** "Bot startup complete and synchronized. Now operational."

---

## Phase 2: The Main Trading Loop (The Heartbeat)

This loop runs continuously, representing the core operational cycle of the bot.

1. **Fetch Processed Data:** The loop calls `market_data_handler.get_latest_data()`.
2. **Check for New Information:** If `None`, the loop pauses and waits.
3. **Check Exit Conditions First:** If an exit is triggered, it proceeds to **Phase 4**.
4. **Check for New Entry Signals:** If a valid signal exists, it proceeds to **Phase 3**.

---

## Phase 3: New Trade Workflow (with Hybrid Mode CLI Menu)

This workflow is triggered by a valid, non-zero signal when all entry conditions are met.

1. **Check Entry Rules:** The `LiveTradingSessionManager` checks if a new trade can be opened.
2. **Calculate Trade Proposal:** `trade_execution_engine.calculate_entry_details()` is called.
3. **Hybrid Mode Check:**  
    * If operating in **automatic mode**:  
        - The trade is executed immediately.
    * If operating in **hybrid mode**:  
        - The bot presents a CLI menu to the user, displaying all trade details (direction, price, quantity, SL, TP, etc.).
        - The CLI menu supports:
            - `y` or `yes`: Approve the trade and proceed with execution.
            - `n` or `no`: Reject the trade; no entry is made.
            - `close`: Close the current open position (if any).
            - `close X`: Close fraction `X` (e.g., `close 0.5`) of the current open position.
            - `help`: Print trade details and menu options again.
            - `skip`: Skip this trade and move to next cycle.
        - **Timeout Handling:** If no input is received within the configured timeout period (e.g., 60 seconds), the trade is skipped and logged.
        - **Input Validation:** Invalid responses are reprompted; only accepted commands are processed.
        - The bot acts according to the user's choice.
4. **Execute & Verify Entry:** The `TradeCycleProcessor` executes the robust entry sequence, ensuring every order placed (entry, SL, TP) is tracked by its unique order ID.
5. **Reconcile & Persist:** The final, verified position is created and saved to the `LiveTradingSessionManager` and the `state.json` file.
6. **Send Notification:** "TRADE ENTERED: [Details: Direction, Price, Qty, SL, TP]"

---

## Phase 4: Close Position Workflow

This is triggered by an exit condition (SL, TP, max holding, reversal, or manual close).

1. **Automatic & Hybrid Mode:**  
    - All exits triggered by SL, TP, liquidation, reversal signal, or max holding are executed immediately—**no confirmation is required in hybrid mode**.
    - Manual closes (full or partial) can be triggered by the CLI menu in hybrid mode.
2. **Cancel Inactive Orders:** The `TradeExecutionEngine` cancels the remaining SL or TP order.
3. **Execute the Close:** The `TradeCycleProcessor` places the closing market order (full or partial as requested).
4. **Finalize & Persist:** The final PnL is calculated, and the state is updated and saved.
5. **Send Notification:** "TRADE CLOSED: [Details: Direction, Price, PnL, Reason]"

---

## Phase 5: Graceful Shutdown

This is triggered by the shutdown handler (e.g., Ctrl+C).

1. The main trading loop is stopped.
2. The bot initiates the **Phase 4: Close Position Workflow** for any open position.
3. A final `save_bot_state()` call is made.
4. **Send Notification:** "Bot has shut down gracefully."

---

### **Assumptions & Caveats**

- This workflow assumes **exclusive control over the trading symbol/account**. No manual trades or other bots should place trades for this symbol while the bot is running.
- The bot relies solely on its internal state (tracked order IDs and positions) and cancels/closes any position or order it does not recognize.
- In **hybrid mode**, the bot waits for terminal input for every trade entry. If no input is received (e.g., unattended), trades will not be executed. Manual/partial closes can be triggered via the CLI menu.
- All exit conditions (SL, TP, max holding, reversal) are executed automatically, regardless of mode.
- **SL/TP protection is always enforced:** At startup, the bot will reconstruct missing SL/TP orders automatically using `TradeCycleProcessor.ensure_sltp_orders()` before entering main loop.

---