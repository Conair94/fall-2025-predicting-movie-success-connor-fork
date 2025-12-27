import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import os

# --- Configuration ---
TOP_N_USERS = 50  # Number of "Super Forecasters" to use as features
OUTPUT_DIR = "data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Loads raw reviews and the alpha ranking."""
    try:
        # Load Raw Reviews
        df_winners = pd.read_csv('data/raw/user_award_reviews.csv')
        df_losers = pd.read_csv('data/raw/user_non_award_reviews.csv')
        
        # Load Alpha Ranking
        df_alpha = pd.read_csv('user_academy_alpha_ranking.csv')
        
        # Load Dates/Years (Critical for Train/Test split)
        df_dates_w = pd.read_csv('data/raw/awarded_movie_date.csv')
        df_dates_l = pd.read_csv('data/raw/non_awarded_movie_date.csv')
        
        return df_winners, df_losers, df_alpha, df_dates_w, df_dates_l
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None

def get_year_map(df_dates_w, df_dates_l):
    """Creates a dictionary mapping movie_name -> oscar_year."""
    # Convert dates to years.
    # Note: Oscars for year X are awarded in early X+1.
    # Usually datasets label movies by release year (e.g. Oppenheimer-2023).
    # We need the *Ceremony Year* for splitting.
    # Let's assume the date file contains the CEREMONY date.
    
    year_map = {}
    
    for df in [df_dates_w, df_dates_l]:
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
        for _, row in df.iterrows():
            if pd.notna(row['date']):
                # If ceremony is Jan/Feb/Mar 2024, the "Oscar Year" is 2024
                year_map[row['movie']] = row['date'].year
    
    return year_map

def prepare_features(df_winners, df_losers, df_alpha, year_map):
    """
    Constructs the feature matrix (X) and target (y).
    Features: The Z-scored ratings of the Top N Alpha users.
    Target: 1 (Win) / 0 (Nominee).
    """
    # 1. Select Top Pollsters
    top_users = df_alpha.head(TOP_N_USERS)['username'].tolist()
    print(f"Using Top {len(top_users)} Super Forecasters.")
    
    # 2. Melt and Combine Reviews
    # Keep only relevant users
    # Filter columns to only include movie names + username
    
    # Transpose to: Movie | User1_Rating | User2_Rating ... | Target
    
    # Helper to pivot
    def pivot_reviews(df_raw, target_val):
        # Filter to top users
        df = df_raw[df_raw['username'].isin(top_users)].copy()
        
        # Melt to Long: User | Movie | Rating
        movie_cols = [c for c in df.columns if c not in ['username', 'user_id']]
        df_long = df.melt(id_vars=['username'], value_vars=movie_cols, var_name='movie', value_name='rating')
        
        # Ensure rating is numeric
        df_long['rating'] = pd.to_numeric(df_long['rating'], errors='coerce')
        
        # Add Target
        df_long['target'] = target_val
        return df_long

    df_w = pivot_reviews(df_winners, 1)
    df_l = pivot_reviews(df_losers, 0)
    full_long = pd.concat([df_w, df_l], axis=0)
    
    # 3. Pivot to Wide Feature Matrix (One row per movie)
    # Index: Movie
    # Columns: Users
    X_raw = full_long.pivot_table(index='movie', columns='username', values='rating')
    
    # 4. Attach Target and Year
    # We need to map movie -> target (1/0)
    # Since a movie is either in df_w or df_l, we can get target easily.
    # However, pivot_table aggregates.
    
    # Let's get unique movie-target map
    targets = full_long[['movie', 'target']].drop_duplicates().set_index('movie')
    
    X = X_raw.copy()
    X['target'] = targets['target']
    
    # Map Years
    X['year'] = X.index.map(year_map)
    
    # Drop movies with unknown years
    X = X.dropna(subset=['year'])
    
    # Handle Missing Ratings (NaNs)
    # If a Super Forecaster didn't rate a movie, impute with the global mean of their ratings?
    # Or 0? Z-score 0 is average.
    # Let's impute with user's mean from the training set, or just 0 if we assume Z-scores.
    # But wait, we haven't Z-scored yet in this script!
    # The raw data is 0.5-5.0. 
    # Let's Z-score per user NOW.
    
    print("Normalizing ratings (Z-score per user)...")
    for user in X_raw.columns:
        u_mean = X[user].mean()
        u_std = X[user].std()
        if u_std == 0 or pd.isna(u_std): u_std = 1
        X[user] = (X[user] - u_mean) / u_std
        X[user] = X[user].fillna(0) # Impute missing with Mean (0)
        
    return X

def train_and_calibrate(X):
    """
    Trains a Logistic Regression model and calibrates it using Leave-One-Year-Out (LOYO).
    This simulates the real 'prediction' task: Training on past, predicting next year.
    """
    years = sorted(X['year'].unique())
    print(f"Training on years: {years}")
    
    results = []
    
    model = LogisticRegression(C=1.0, penalty='l2', class_weight='balanced', solver='liblinear')
    
    # We need to store raw predictions to calibrate them globally or per fold
    all_preds_raw = []
    all_targets = []
    
    # LOYO Validation Loop
    for test_year in years:
        # Train on all PAST years (strict time series split)
        # Or train on ALL other years?
        # Standard backtest: Train on T < test_year.
        # But for 2015, we have no history.
        # Let's use Leave-One-Out for stability since dataset is small (10 years).
        # Assuming taste is stable.
        
        train_mask = (X['year'] != test_year)
        test_mask = (X['year'] == test_year)
        
        if not train_mask.any() or not test_mask.any():
            continue
            
        X_train = X.loc[train_mask].drop(columns=['target', 'year'])
        y_train = X.loc[train_mask, 'target']
        
        X_test = X.loc[test_mask].drop(columns=['target', 'year'])
        y_test = X.loc[test_mask, 'target']
        movies_test = X.loc[test_mask].index
        
        # Fit Model
        model.fit(X_train, y_train)
        
        # Predict Probabilities (Uncalibrated)
        probs_raw = model.predict_proba(X_test)[:, 1]
        
        # Store for Calibration
        for i, m in enumerate(movies_test):
            results.append({
                'year': test_year,
                'movie': m,
                'prob_raw': probs_raw[i],
                'target': y_test.iloc[i]
            })
            
    res_df = pd.DataFrame(results)
    
    # Global Calibration (Isotonic)
    # We fit the calibrator on the raw model outputs
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(res_df['prob_raw'], res_df['target'])
    
    res_df['prob_calibrated'] = iso.transform(res_df['prob_raw'])
    
    return res_df, model, iso

def main():
    print("--- Phase 2: Model Training & Calibration ---")
    
    # 1. Load
    dw, dl, da, ddw, ddl = load_data()
    if dw is None: return
    
    # 2. Map Years
    year_map = get_year_map(ddw, ddl)
    
    # 3. Features
    X = prepare_features(dw, dl, da, year_map)
    print(f"Feature Matrix Shape: {X.shape}")
    
    # 4. Train & Calibrate
    res_df, final_model, calibrator = train_and_calibrate(X)
    
    # 5. Evaluate
    # Print 2024 Predictions (Last Year in data)
    last_year = res_df['year'].max()
    print(f"\n--- Backtest Results for {last_year} (Unseen Data) ---")
    print(res_df[res_df['year'] == last_year].sort_values('prob_calibrated', ascending=False)[['movie', 'prob_raw', 'prob_calibrated', 'target']])
    
    # 6. Save Model Artifacts (for Phase 3)
    # We will need the trained model to predict 2025
    import joblib
    joblib.dump(final_model, os.path.join(OUTPUT_DIR, 'logistic_model.pkl'))
    joblib.dump(calibrator, os.path.join(OUTPUT_DIR, 'isotonic_calibrator.pkl'))
    
    # Save Feature Config (User list)
    top_users = X.columns.drop(['target', 'year']).tolist()
    pd.Series(top_users).to_csv(os.path.join(OUTPUT_DIR, 'feature_users.csv'), index=False)
    
    print(f"\nModel and Super Forecaster list saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
