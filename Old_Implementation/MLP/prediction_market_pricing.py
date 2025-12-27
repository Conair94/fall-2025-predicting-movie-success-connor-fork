import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import argparse
import sys

# --- Data Processing Functions (Adapted from MLP.ipynb) ---

def sanatize(df):
    cols = df.columns.difference(['username'])
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

def user_intersection(df_1, df_2):
    user_col = 'username'
    common = set(df_1[user_col]) & set(df_2[user_col])

    a = (df_1[df_1[user_col].isin(common)]
         .sort_values(user_col)
         .drop_duplicates(subset=[user_col], keep="first")
         .reset_index(drop=True))
    b = (df_2[df_2[user_col].isin(common)]
         .sort_values(user_col)
         .drop_duplicates(subset=[user_col], keep="first")
         .reset_index(drop=True))

    a.insert(0, "user_id", range(len(a)))
    b.insert(0, "user_id", range(len(b)))

    assert len(a) == len(b), "still mismatched — check for NaNs or types"
    return a, b

def add_mean_std(df, user_rating_counts):
    # This logic was slightly implicit in the notebook, adapting here based on notebook logic
    # In notebook: add_mean_std(user_rating_counts) is called.
    values = np.arange(0.5, 5.5, 0.5)
    count_cols = [str(x) for x in values]
    vals = pd.Series(values, index=count_cols)
    
    # Ensure columns exist and are numeric
    for c in count_cols:
        if c not in df.columns:
            df[c] = 0
    
    n = df[count_cols].sum(axis=1)
    # Avoid division by zero
    n = n.replace(0, 1) 
    
    df['mean'] = (df[count_cols] @ vals) / n
    m2 = (df[count_cols] @ (vals**2)) / n
    var_pop = m2 - df['mean']**2
    # simple fix for negative variance due to float precision
    var_pop = var_pop.clip(lower=0)
    df['std'] = np.sqrt(var_pop * n/(n-1))
    df['std'] = df['std'].fillna(1.0) # Fallback

def change_raw_rating_to_z_score(df, user_rating_counts):
    n_rows, n_cols = df.shape
    # Determine which columns are movie ratings (skip user_id, username)
    # The notebook assumes columns 2 onwards are movies
    
    # We need efficient lookup
    means = user_rating_counts['mean']
    stds = user_rating_counts['std']
    
    # It allows inplace modification.
    # To be safe and faster, we can use vectorized operations if possible, 
    # but let's stick to the notebook's iterative approach for fidelity, or optimize slightly.
    
    # Iterating over dataframe is slow, but consistent with original code.
    # Let's try to do it slightly better but safely. 
    
    for r in range(n_rows):
        user = df.iat[r, 1] # username
        if user not in means.index:
            continue
            
        mean = means.at[user]
        std = stds.at[user]
        if pd.isna(std) or std == 0:
            std = 1.0
            
        for c in range(2, n_cols):
            rating = df.iat[r, c]
            if not pd.isna(rating):
                df.iat[r, c] = np.clip((rating - mean)/std, -4, 4)

def movie_range(df, time_frame):
    start, end = time_frame
    # Check format. The CSV likely has MM/DD/YYYY
    df['date'] = pd.to_datetime(
        df['date'], format='%m/%d/%Y', errors='coerce'
    )
    # Fallback for mixed formats if any (optional)
    
    mask = df['date'].dt.year.between(start, end, inclusive='both')
    return df.loc[mask, 'movie'].tolist()

def combine_reviews(award_reviews, non_award_reviews):
    aw = award_reviews.set_index(list(award_reviews.columns[:2]))
    naw = non_award_reviews.set_index(list(non_award_reviews.columns[:2]))
    idx = aw.index.union(naw.index)
    aw, naw = aw.reindex(idx), naw.reindex(idx)
    return pd.concat([aw, naw], axis=1)

def build_bag_of_users(reviews_z, movies, y):
    # returns X_x, X_m, y_vec
    missing = [m for m in movies if m not in reviews_z.columns]
    # In a real pricing scenario, we might have movies in the date file but not in reviews yet?
    # For now, we warn or fail. The notebook raises error.
    if missing:
        print(f"Warning: Movies missing from review data: {missing}")
        movies = [m for m in movies if m in reviews_z.columns]
    
    if not movies:
        return None, None, None, []

    Z = reviews_z[movies].astype("float32").to_numpy(copy=True)
    M = ~np.isnan(Z)
    X = np.where(M, Z, 0.0).astype("float32")
    
    y_vec = np.array([int(y.get(m, 0)) for m in movies], dtype=np.int64)
    return X.T.astype("float32"), M.T.astype("float32"), y_vec, movies

def main():
    parser = argparse.ArgumentParser(description="Predict Oscar success probabilities for a target year.")
    parser.add_argument('target_year', type=int, help='The year to predict (e.g., 2024)')
    args = parser.parse_args()

    target_year = args.target_year
    
    print(f"--- Loading Data ---")
    try:
        award_reviews = pd.read_csv('user_award_reviews.csv')
        non_award_reviews = pd.read_csv('user_non_award_reviews.csv')
        user_rating_counts = pd.read_csv('user_rating_counts.csv')
        awards_date = pd.read_csv('awarded_movie_date.csv')
        non_awards_date = pd.read_csv('non_awarded_movie_date.csv')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        print("Please ensure you are in the 'MLP' directory or files exist.")
        sys.exit(1)

    # 1. Preprocess Users
    print("Processing user data...")
    award_reviews, non_award_reviews = user_intersection(award_reviews, non_award_reviews)
    
    sanatize(award_reviews)
    sanatize(non_award_reviews)
    sanatize(user_rating_counts)
    
    add_mean_std(user_rating_counts, None) # user_rating_counts is modified in place
    user_rating_counts.set_index('username', inplace=True)
    
    change_raw_rating_to_z_score(award_reviews, user_rating_counts)
    change_raw_rating_to_z_score(non_award_reviews, user_rating_counts)
    
    # 2. Setup Movies and Target
    print(f"Configuring years. Target Year: {target_year}")
    
    # Define Training range: Start from 2015 (min in notebook) up to target_year - 1
    # We can inspect data to find absolute min, but 2015 is safe based on notebook.
    training_start = 2015
    training_end = target_year - 1
    
    if training_end < training_start:
        print("Target year is too early. Need prior years for training.")
        sys.exit(1)

    print(f"Training Range: {training_start}-{training_end}")
    
    train_movies = movie_range(awards_date, (training_start, training_end)) + \
                   movie_range(non_awards_date, (training_start, training_end))
    
    target_movies_list = movie_range(awards_date, (target_year, target_year)) + \
                         movie_range(non_awards_date, (target_year, target_year))
    
    if not train_movies:
        print("No training movies found in range.")
        sys.exit(1)
    if not target_movies_list:
        print(f"No movies found for target year {target_year}.")
        sys.exit(1)

    # 3. Combine Reviews
    reviews_z = combine_reviews(award_reviews, non_award_reviews)
    
    # 4. Prepare Target Dict (y)
    # We assume 'award_reviews' columns (from index 2) are winners?
    # The notebook says: y = {movie : 1 for movie in award_reviews.columns[2:]} | {movie : 0 ...}
    # But wait, 'award_reviews' was filtered.
    # We should reconstruct the global truth from the source files or column names.
    # Actually, simpler: check if movie is in 'awards_date'.
    # Note: notebook logic relies on the columns present in the review csvs.
    
    # Re-reading original columns to be sure of class 
    # (Since we modified DFs, let's rely on movie lists from date files for labels)
    
    # In the notebook:
    # y = {movie : 1 for movie in award_reviews.columns[2:]} | {movie : 0 for movie in non_award_reviews.columns[2:]}
    # This implies the CSVs are split by outcome.
    
    # Let's recreate y map based on the original data loading (before processing/filtering might remove cols?)
    # Actually, 'user_intersection' drops rows (users), not columns (movies).
    # So the columns are preserved.
    
    y = {}
    # award_reviews columns [2:] are winners
    for m in award_reviews.columns[2:]:
        y[m] = 1
    # non_award_reviews columns [2:] are losers
    for m in non_award_reviews.columns[2:]:
        y[m] = 0
        
    # Also cross-reference with our date lists just in case
    # If a movie is in 'train_movies' but not in y, we can't train on it.
    
    # 5. Build Datasets
    print("Building datasets...")
    Xtr_x, Xtr_m, ytr, final_train_movies = build_bag_of_users(reviews_z, train_movies, y)
    Xte_x, Xte_m, yte, final_target_movies = build_bag_of_users(reviews_z, target_movies_list, y)
    
    if Xtr_x is None or Xte_x is None:
        print("Failed to build dataset (missing movies in review data).")
        sys.exit(1)

    # 6. Preprocessing (Scaling)
    # Notebook: scales Xtr_x, then concatenates mask.
    scaler = StandardScaler()
    Xtr_x_scaled = scaler.fit_transform(Xtr_x)
    Xte_x_scaled = scaler.transform(Xte_x) # Transform target using train scaler
    
    Xtr = np.concatenate([Xtr_x_scaled, Xtr_m], axis=1).astype(np.float32)
    Xte = np.concatenate([Xte_x_scaled, Xte_m], axis=1).astype(np.float32)
    
    # 7. Train MLP
    print(f"Training MLP on {len(final_train_movies)} movies...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128,), 
        random_state=0,
        max_iter=1000,
        learning_rate_init=1e-3,
        alpha=1e-4,
        early_stopping=False,
        shuffle=True,
        verbose=False,
    )
    clf.fit(Xtr, ytr)
    
    # 8. Predict
    print(f"Predicting for {len(final_target_movies)} movies in {target_year}...")
    probs = clf.predict_proba(Xte)[:, 1]
    
    # 9. Output Market Prices
    results = pd.DataFrame({
        'Movie': final_target_movies,
        'Raw_Prob': probs
    })
    
    results.sort_values('Raw_Prob', ascending=False, inplace=True)
    
    print("\n--- PREDICTION MARKET PRICES ---")
    print(f"Target Year: {target_year}\n")
    print(f"{ 'Movie':<40} | {'Raw Price ($)':<15} | {'Norm Price ($)':<15}")
    print("-" * 75)
    
    total_prob = results['Raw_Prob'].sum()
    
    for _, row in results.iterrows():
        raw_p = row['Raw_Prob']
        norm_p = raw_p / total_prob if total_prob > 0 else 0.0
        print(f"{row['Movie']:<40} | ${raw_p:.4f}        | ${norm_p:.4f}")

    print("\nNote: 'Raw Price' assumes independent binary outcomes.")
    print("'Norm Price' assumes a single winner (mutually exclusive) among the listed movies.")

if __name__ == "__main__":
    main()
