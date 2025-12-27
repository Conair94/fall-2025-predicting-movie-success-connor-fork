import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import argparse

# --- Configuration ---
RISK_FREE_RATE = 0.045 # 4.5%
OSCAR_DATE_2025 = "2025-03-02" # 97th Oscars
OSCAR_DATE_2026 = "2026-03-15" # 98th Oscars (Hypothetical)

def load_artifacts():
    try:
        model = joblib.load('data/processed/logistic_model.pkl')
        calibrator = joblib.load('data/processed/isotonic_calibrator.pkl')
        return model, calibrator
    except FileNotFoundError:
        print("Model artifacts not found. Run training first.")
        return None, None

def get_discount_factor(target_date_str):
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    today = datetime.now()
    
    if target_date < today:
        return 1.0 # Already happened
        
    days_to_maturity = (target_date - today).days
    years = days_to_maturity / 365.0
    
    # Continuous discounting: e^(-rt)
    # or Discrete: 1 / (1+r)^t
    df = 1 / ((1 + RISK_FREE_RATE) ** years)
    
    print(f"Time to Maturity: {days_to_maturity} days ({years:.2f} years)")
    print(f"Discount Factor (r={RISK_FREE_RATE:.1%}): {df:.4f}")
    return df

def run_pricing(year):
    print(f"--- Running Pricing Engine for Oscars {year} ---")
    
    # 1. Load Model
    model, calibrator = load_artifacts()
    if not model: return

    # 2. Load Market Odds (Scraped)
    # We assume 'scrape_polymarket.py' has run and produced 'data/raw/market_odds_polymarket.csv'
    # Or 'market_odds.csv' manually.
    # For now, we will Mock the market data if file doesn't exist, to demonstrate the logic. 
    
    market_file = "data/raw/market_odds_polymarket.csv"
    if os.path.exists(market_file):
        df_market = pd.read_csv(market_file)
        # Filter for year/event? The scraper should have filtered.
        # We need to map 'Movie Name' to our internal format (lowercase, hyphens)
        # Polymarket names: "Oppenheimer", "Barbie"
        # Internal: "oppenheimer-2023", "barbie"
        # We need a Fuzzy Matcher or manual map.
        print("Loaded Real Market Data.")
    else:
        print("Market Data not found. Mocking data for demonstration.")
        # Mock Data for 2026
        df_market = pd.DataFrame({
            'movie': ['Project Hail Mary', 'Dune Messiah', 'The Movie Critic', 'Generic Biopic'],
            'price': [0.15, 0.25, 0.10, 0.05]
        })

    # 3. Load User Reviews (The Alpha Source)
    # In a real run, we would scrape FRESH reviews for these specific market candidates.
    # Since we can't scrape right now, we assume we have a feature matrix `X_live` ready.
    # For this script, I will generate random "Super Forecaster" ratings for the market movies
    # to show the pipeline output.
    
    print("Generating Model Predictions (Simulated on Fresh Data)...")
    
    # Simulate Model Output (Raw Probability)
    # In production: X = build_features(users, movies) -> model.predict_proba(X)
    # Here: Random for demo
    np.random.seed(42)
    raw_probs = np.random.uniform(0.1, 0.9, size=len(df_market))
    
    # Calibrate
    cal_probs = calibrator.transform(raw_probs)
    
    # Normalize (Mutually Exclusive)
    norm_probs = cal_probs / np.sum(cal_probs)
    
    df_market['model_prob_win'] = norm_probs
    
    # 4. Apply Time Value of Money
    target_date = OSCAR_DATE_2026 if str(year) == "2026" else OSCAR_DATE_2025
    df = get_discount_factor(target_date)
    
    # Fair Price = Probability * Discount Factor
    # This is the max price we should pay today for a $1 payout.
    df_market['fair_price_bid'] = df_market['model_prob_win'] * df
    
    # 5. Calculate Edge
    # Edge = Fair Price - Market Ask Price
    # If Fair Price > Market Price, we buy.
    df_market['edge'] = df_market['fair_price_bid'] - df_market['price']
    
    # 6. Kelly Stake
    # Simple proportional: Bet size ~ Edge
    bankroll = 10000
    df_market['kelly_bet'] = df_market.apply(lambda x: 0 if x['edge'] <= 0 else (x['edge'] * 0.5 * bankroll), axis=1)
    
    # Output
    print("\n--- PRICING SHEET ---")
    print(df_market[['movie', 'price', 'model_prob_win', 'fair_price_bid', 'edge', 'kelly_bet']].sort_values('edge', ascending=False))
    
    # Save
    df_market.to_csv(f"data/processed/pricing_sheet_{year}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, default=2026)
    args = parser.parse_args()
    run_pricing(args.year)
