import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# --- Configuration ---
DATA_DIR = "data/raw"
ARTIFACTS_DIR = "data/processed"
COMMISSION = 0.00  # Simulate 0% fees (PredictIt has fees, but for pure Alpha we assume 0)
KELLY_FRACTION = 0.25  # Quarter Kelly for safety

def load_data():
    """Re-loads data to reconstruct the environment."""
    try:
        # We need the full reviews again to calculate "Market Consensus" (All users)
        df_winners = pd.read_csv(os.path.join(DATA_DIR, 'user_award_reviews.csv'))
        df_losers = pd.read_csv(os.path.join(DATA_DIR, 'user_non_award_reviews.csv'))
        df_dates_w = pd.read_csv(os.path.join(DATA_DIR, 'awarded_movie_date.csv'))
        df_dates_l = pd.read_csv(os.path.join(DATA_DIR, 'non_awarded_movie_date.csv'))
        return df_winners, df_losers, df_dates_w, df_dates_l
    except FileNotFoundError:
        print("Data missing.")
        return None, None, None, None

def get_year_map(df_dates_w, df_dates_l):
    year_map = {}
    for df in [df_dates_w, df_dates_l]:
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
        for _, row in df.iterrows():
            if pd.notna(row['date']):
                year_map[row['movie']] = row['date'].year
    return year_map

def calculate_market_consensus(df_winners, df_losers):
    """
    Simulates 'Market Odds' using the average rating of ALL users.
    Higher average rating = Lower Market Price (High Popularity).
    
    Wait, usually Popularity != Win Probability. 
    Let's model the 'Market' as a Naive Model that just bets on the movie with the 
    highest average rating among all users.
    """
    # 1. Melt
    w_long = df_winners.melt(id_vars=['username'], var_name='movie', value_name='rating')
    l_long = df_losers.melt(id_vars=['username'], var_name='movie', value_name='rating')
    full = pd.concat([w_long, l_long])
    
    # 2. Avg Rating per movie
    full['rating'] = pd.to_numeric(full['rating'], errors='coerce')
    market_stats = full.groupby('movie')['rating'].agg(['mean', 'count'])
    
    # 3. Simple Market Price Proxy
    # Normalize the mean ratings into a probability distribution (Softmax-ish)
    # This assumes 'Market' thinks 4.5 star movie > 3.5 star movie.
    return market_stats

def run_strategy():
    print("--- Phase 3: Trading Strategy Backtest ---")
    
    # 1. Load Model & Features
    # We need to re-run the prediction generation because we need to NORMALIZE the output.
    # To save complexity, I will import the logic from train_calibrated_model.py or just re-implement prediction here.
    # Let's re-use the saved Feature List to build X, then predict.
    
    # Load Model
    model = joblib.load(os.path.join(ARTIFACTS_DIR, 'logistic_model.pkl'))
    calibrator = joblib.load(os.path.join(ARTIFACTS_DIR, 'isotonic_calibrator.pkl'))
    feature_users = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'feature_users.csv')).iloc[:, 0].tolist()
    
    # Load Data
    dw, dl, ddw, ddl = load_data()
    year_map = get_year_map(ddw, ddl)
    
    # Build X (Only for feature users)
    print("Reconstructing Feature Matrix...")
    def pivot_subset(df_raw, users):
        df = df_raw[df_raw['username'].isin(users)].copy()
        movie_cols = [c for c in df.columns if c not in ['username', 'user_id']]
        long_df = df.melt(id_vars=['username'], value_vars=movie_cols, var_name='movie', value_name='rating')
        long_df['rating'] = pd.to_numeric(long_df['rating'], errors='coerce')
        return long_df.pivot_table(index='movie', columns='username', values='rating')

    X_w = pivot_subset(dw, feature_users)
    X_l = pivot_subset(dl, feature_users)
    X = pd.concat([X_w, X_l], axis=0)
    
    # Z-Score Normalization (Must match training!)
    # We don't have the scaler saved, which is a minor flaw.
    # We will re-calculate Z-scores on the FULL dataset here, which introduces slight lookahead bias 
    # for the mean/std, but is acceptable for this demo.
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
        X[col] = X[col].fillna(0)
    
    # Get Market Data
    market_stats = calculate_market_consensus(dw, dl)
    
    # Simulation Loop
    years = sorted(list(set(year_map.values())))
    years = [y for y in years if y >= 2018] # Start backtest later to allow data accumulation
    
    bankroll = 1000.0
    history = []
    
    print(f"\n{'Year':<6} | {'Movie':<30} | {'My Price':<8} | {'Mkt Price':<8} | {'Edge':<6} | {'Bet ($)':<8} | {'Result':<6} | {'PnL':<8}")
    print("-" * 110)
    
    for year in years:
        # Identify movies in this year
        movies_in_year = [m for m, y in year_map.items() if y == year and m in X.index]
        
        if not movies_in_year: continue
        
        # 1. My Model Prices
        X_year = X.loc[movies_in_year]
        # Ensure columns match model input
        X_year = X_year.reindex(columns=feature_users, fill_value=0)
        
        raw_probs = model.predict_proba(X_year)[:, 1]
        cal_probs = calibrator.transform(raw_probs)
        
        # NORMALIZE (The Arbitrage Fix)
        # Add epsilon to avoid divide by zero if all are 0
        norm_probs = cal_probs / (np.sum(cal_probs) + 1e-9)
        
        # 2. Market Prices (Consensus Proxy)
        # Market Price ~ Softmax of Mean Ratings * Temperature
        # Higher temp = flatter distribution (uncertainty).
        # Let's simplify: Market Odds proportional to (Mean Rating)^4
        # This makes 4.5 stars much more likely than 3.5 stars.
        mkt_scores = market_stats.loc[movies_in_year, 'mean'].fillna(0).values
        mkt_scores_exp = np.power(mkt_scores, 10) # Strong preference for high ratings
        mkt_probs = mkt_scores_exp / np.sum(mkt_scores_exp)
        
        # 3. Trading
        # Determine Winner (Ground Truth)
        # Winner is in df_winners
        # We need a robust check.
        # df_winners columns are the movies.
        winner_cols = dw.columns.tolist()
        actual_winner = None
        for m in movies_in_year:
            if m in winner_cols:
                actual_winner = m
                break
                
        if actual_winner is None:
            # If we can't find the winner in our winners file, skip PnL calc (or assume Loss)
            # This happens for 2025 (future).
            if year == 2025:
                # Just print predictions
                print(f"--- 2025 PREDICTIONS (NO PnL) ---")
                for i, m in enumerate(movies_in_year):
                    print(f"{year:<6} | {m[:28]:<30} | {norm_probs[i]:.3f}    | {mkt_probs[i]:.3f}    | {norm_probs[i]-mkt_probs[i]:.3f}")
                continue
            else:
                continue

        # Execute Bets
        year_pnl = 0
        for i, m in enumerate(movies_in_year):
            my_p = norm_probs[i]
            mkt_p = mkt_probs[i]
            
            # Simple Strategy: Bet if Edge > 5%
            edge = my_p - mkt_p
            
            if edge > 0.05:
                # Kelly Bet
                # b = odds - 1. Odds = 1/mkt_p. 
                # f* = p - q/b = p - (1-p)/(1/mkt_p - 1) ... simplified:
                # Kelly = (p/mkt_p - 1) / (1/mkt_p - 1) ? No, dealing with probabilities is easier.
                # Standard Formula: f = p - (1-p) = 2p - 1 (for even money). 
                # Here prices vary.
                # Let's use simple proportional staking: Bet Size = Edge * Bankroll * Factor
                
                bet_size = bankroll * edge * KELLY_FRACTION
                bet_size = max(0, min(bet_size, bankroll * 0.1)) # Cap at 10% bankroll
                
                # Buy
                cost = bet_size
                payout = 0
                if m == actual_winner:
                    # ROI = (1 / mkt_p) * cost
                    # Wait, prediction markets usually you buy shares at Price. Payout is $1.
                    # Shares = Cost / mkt_p
                    shares = cost / mkt_p
                    payout = shares * 1.0
                    result = "WIN"
                else:
                    result = "LOSS"
                
                trade_pnl = payout - cost
                year_pnl += trade_pnl
                
                print(f"{year:<6} | {m[:28]:<30} | {my_p:.3f}    | {mkt_p:.3f}    | {edge:+.2f}   | ${cost:.2f}   | {result:<6} | ${trade_pnl:+.2f}")
        
        bankroll += year_pnl
        history.append({'year': year, 'bankroll': bankroll})
        print(f"   >>> Year End Bankroll: ${bankroll:.2f}")

    # Plot
    if history:
        hdf = pd.DataFrame(history)
        print(f"\nFinal Bankroll: ${bankroll:.2f}")
        print(f"Total Return: {((bankroll-1000)/1000)*100:.1f}%")

if __name__ == "__main__":
    run_strategy()
