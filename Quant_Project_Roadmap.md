# Quant Project Roadmap: Oscar Prediction Market Alpha

## Objective
Transform the project from a black-box Oscar predictor into a Quantitative Research (QR) portfolio piece focused on **Signal Combination**, **Probability Calibration**, and **Expected Value (EV) Extraction**.

## Phase 1: Feature Engineering (The "Pollster" Analysis)
**Goal:** Identify and weight "Super Forecaster" users based on historical predictive power ("Academy Alpha"), rather than raw activity.

1.  **Data Processing:**
    *   Standardize all review data (Z-scores per user to handle "grade inflation" bias).
    *   Define the Universe: "Oscar Contenders" (Winners + Nominees).

2.  **The "Academy Alignment" Score (Rolling Window):**
    *   Treat each user as a "Pollster".
    *   **Metric:** Calculate the **AUC (Area Under Curve)** or **Information Coefficient (IC)** for each user relative to Oscar outcomes.
    *   *Question:* Does User X consistently rate Best Picture winners higher than nominees?
    *   **Window:** Calculate scores using only data available prior to the prediction year (e.g., score for 2024 is based on 2015-2023 history).

3.  **Alpha Buckets:**
    *   Cluster users into "Factor" groups:
        *   **Momentum:** Users who align with popular consensus.
        *   **Contrarian/Smart Money:** Users who correctly identified "upset" winners (e.g., *Green Book*, *CODA*) when the crowd was wrong.

## Phase 2: The Prediction Model (Calibration)
**Goal:** Estimate robust win probabilities $P(Win)$, not just binary outcomes.

1.  **Model Selection:**
    *   **Logistic Regression:** Good baseline, interpretable weights (e.g., "Smart Money" ratings * 1.5 + "Crowd" ratings * 0.5).
    *   **Gradient Boosting (XGBoost):** Captures non-linear interactions (e.g., "High rating from Smart Money matters MORE if the movie is also a Drama").

2.  **Calibration:**
    *   Apply **Isotonic Regression** to ensure model outputs represent true probabilities.
    *   *Verification:* A calibration plot (Reliability Diagram). If the model predicts 70%, the event should happen 70% of the time.

## Phase 3: The Trading Strategy (Alpha Extraction)
**Goal:** Demonstrate "Edge" against a market.

1.  **Market Simulation:**
    *   Mock "Market Odds" using:
        *   Historical Betting Odds (if available).
        *   Naive Consensus (e.g., average rating of *all* users).
    
2.  **Strategy Implementation:**
    *   **Long/Short:** Bet Long if $P_{Model} > P_{Market}$. Bet Short (or do not bet) if $P_{Model} < P_{Market}$.
    *   **Kelly Criterion:** Size bets based on the magnitude of the edge.

3.  **Backtesting:**
    *   Calculate **Sharpe Ratio** and **Max Drawdown** of the strategy, not just "Accuracy".
    *   Show that the strategy generates profit even in years where the "Winner" wasn't the #1 model pick, by avoiding bad bets on overhyped favorites.

## Phase 4: Evaluation & Visualization
*   **"Pollster" Leaderboard:** Who are the best predictors?
*   **Calibration Curve:** Proof of model robustness.
*   **PnL Cumulative Chart:** The "Equity Curve" of the strategy.
