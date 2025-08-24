# Trading Bot - Complete Operational Workflow

**Author:** DimitriKenne  
**Date:** 2025-08-24 02:40:59 UTC

This document outlines the definitive, end-to-end operational workflow for the live trading bot. It serves as the official blueprint for the `trading_bot.py` implementation.

---

## Phase 1: Initialization and Startup (Robust Reconciliation)

This phase ensures the bot starts in a clean, known, and synchronized state using a comprehensive, two-phase reconciliation process.

1. **Load Configuration:** The bot starts and loads all necessary parameters from the `AppConfig` object, including the unique `bot_id`.
2. **Initialize Core Components:** It creates instances of our main modules (`LifecycleManager`, `TradeCycleProcessor`, etc.).
3. **Send Startup Notification:** "Bot is starting up..."
4. **Connect to Exchange:** The `BinanceFuturesAdapter` connects to the API and fetches initial exchange rules.
5. **Load Persistent State:** The bot loads the last known state (`current_capital`, `open_position`, etc.) from the `state.json` file into the `LiveTradingSessionManager`.

6. **Critical - The Reconciliation Loop:** This is the core of a safe startup.

    *   **Phase A: Fetch Ground Truth**
        1.  Fetch ALL open positions for the trading symbol from the exchange.
        2.  Fetch ALL open orders for the trading symbol from the exchange.

    *   **Phase B: Reconcile State & Enforce Protection**
        1.  **Reconcile Active Position:**
            *   **Match Found:** If the exchange has one position that matches the bot's loaded position (by order ID), it is adopted.
            *   **Orphan Found:** If the exchange has a position but the bot has no state, a `CRITICAL` error is raised, and the bot stops. Manual intervention is required.
            *   **Ghost Found:** If the bot has a position in its state but the exchange does not, a `WARNING` is logged, and the position is cleared from the bot's state.
            *   **Multiple Positions:** If the exchange shows multiple positions, a `WARNING` is logged, and an attempt is made to close the smaller, unexpected positions to consolidate into one.

        2.  **Reconcile SL/TP Orders:**
            *   Based on the now-synced active position, the bot knows what its ideal SL/TP orders should be (by order ID).
            *   **Missing Protection:** If an ideal SL or TP order is *missing* from the exchange, it is placed immediately.
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

## Phase 3: New Trade Workflow (Enhanced with Verification)

This workflow is triggered by a valid, non-zero signal when all entry conditions are met.

1. **Check Entry Rules:** The `LiveTradingSessionManager` checks if a new trade can be opened.
2. **Calculate Trade Proposal:** `trade_execution_engine.calculate_entry_details()` is called.
3. **Execute & Verify Entry:** The `TradeCycleProcessor` executes the robust entry sequence, ensuring every order placed (entry, SL, TP) is tracked by its unique order ID.
4. **Reconcile & Persist:** The final, verified position is created and saved to the `LiveTradingSessionManager` and the `state.json` file.
5. **Send Notification:** "TRADE ENTERED: [Details: Direction, Price, Qty, SL, TP]"

---

## Phase 4: Close Position Workflow (Enhanced with Order Cleanup)

This is triggered by an exit condition.

1. **Cancel Inactive Order:** The `TradeExecutionEngine` cancels the remaining SL or TP order.
2. **Execute the Close:** The `TradeCycleProcessor` places the closing market order.
3. **Finalize & Persist:** The final PnL is calculated, and the state is updated and saved.
4. **Send Notification:** "TRADE CLOSED: [Details: Direction, Price, PnL, Reason]"

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
