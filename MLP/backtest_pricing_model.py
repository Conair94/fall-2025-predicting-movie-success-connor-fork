import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import sys

# --- Helper Functions (Reused) ---

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
    return a, b

def add_mean_std(df):
    values = np.arange(0.5, 5.5, 0.5)
    count_cols = [str(x) for x in values]
    vals = pd.Series(values, index=count_cols)
    for c in count_cols:
        if c not in df.columns: df[c] = 0
    n = df[count_cols].sum(axis=1).replace(0, 1)
    df['mean'] = (df[count_cols] @ vals) / n
    m2 = (df[count_cols] @ (vals**2)) / n
    var_pop = (m2 - df['mean']**2).clip(lower=0)
    df['std'] = np.sqrt(var_pop * n/(n-1)).fillna(1.0)

def change_raw_rating_to_z_score(df, user_rating_counts):
    n_rows, n_cols = df.shape
    means = user_rating_counts['mean']
    stds = user_rating_counts['std']
    for r in range(n_rows):
        user = df.iat[r, 1]
        if user not in means.index: continue
        mean = means.at[user]
        std = stds.at[user]
        if pd.isna(std) or std == 0: std = 1.0
        for c in range(2, n_cols):
            rating = df.iat[r, c]
            if not pd.isna(rating):
                df.iat[r, c] = np.clip((rating - mean)/std, -4, 4)

def movie_range(df, time_frame):
    start, end = time_frame
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    mask = df['date'].dt.year.between(start, end, inclusive='both')
    return df.loc[mask, 'movie'].tolist()

def combine_reviews(award_reviews, non_award_reviews):
    aw = award_reviews.set_index(list(award_reviews.columns[:2]))
    naw = non_award_reviews.set_index(list(non_award_reviews.columns[:2]))
    idx = aw.index.union(naw.index)
    aw, naw = aw.reindex(idx), naw.reindex(idx)
    return pd.concat([aw, naw], axis=1)

def build_dataset(reviews_z, movies, y):
    available_movies = [m for m in movies if m in reviews_z.columns]
    if not available_movies: return None, None, None, []
    
    Z = reviews_z[available_movies].astype("float32").to_numpy(copy=True)
    M = ~np.isnan(Z)
    X = np.where(M, Z, 0.0).astype("float32")
    y_vec = np.array([int(y.get(m, 0)) for m in available_movies], dtype=np.int64)
    return X.T.astype("float32"), M.T.astype("float32"), y_vec, available_movies

# --- Backtest Logic ---

def run_backtest():
    print("--- Starting Backtest (2018 - 2024) ---")
    
    try:
        ar = pd.read_csv('user_award_reviews.csv')
        nar = pd.read_csv('user_non_award_reviews.csv')
        urc = pd.read_csv('user_rating_counts.csv')
        ad = pd.read_csv('awarded_movie_date.csv')
        nad = pd.read_csv('non_awarded_movie_date.csv')
    except Exception as e:
        print(f"Data load failed: {e}")
        return

    # Process Users
    ar, nar = user_intersection(ar, nar)
    sanatize(ar)
    sanatize(nar)
    sanatize(urc)
    add_mean_std(urc)
    urc.set_index('username', inplace=True)
    change_raw_rating_to_z_score(ar, urc)
    change_raw_rating_to_z_score(nar, urc)
    
    y_map = {}
    for m in ar.columns[2:]: y_map[m] = 1
    for m in nar.columns[2:]: y_map[m] = 0
    
    reviews_z = combine_reviews(ar, nar)
    
    # Explicit Ground Truth for Best Picture Winners
    # Keys must match CSV column names exactly
    best_picture_winners = {
        2015: "birdman-or-the-unexpected-virtue-of-ignorance",
        2016: "spotlight",
        2017: "moonlight-2016",
        2018: "the-shape-of-water",
        2019: "green-book",
        2020: "parasite-2019",
        2021: "nomadland",
        2022: "coda-2021",
        2023: "everything-everywhere-all-at-once",
        2024: "oppenheimer-2023"
    }
    
    years_to_test = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    
    print(f"{'Year':<6} | {'Training':<9} | {'Predicted Winner (Highest Prob)':<35} | {'Price':<8} | {'Actual Winner':<30} | {'Result':<6} | {'PnL ($)':<10}")
    print("-" * 120)
    
    cumulative_pnl = 0.0

    for year in years_to_test:
        train_start = 2015
        train_end = year - 1
        
        train_movies = movie_range(ad, (train_start, train_end)) + movie_range(nad, (train_start, train_end))
        test_movies = movie_range(ad, (year, year)) + movie_range(nad, (year, year))
        
        if not train_movies or not test_movies:
            print(f"{year:<6} | Skipped (No Data)")
            continue

        Xtr_x, Xtr_m, ytr, final_train = build_dataset(reviews_z, train_movies, y_map)
        Xte_x, Xte_m, yte, final_test = build_dataset(reviews_z, test_movies, y_map)
        
        if Xtr_x is None or Xte_x is None:
            print(f"{year:<6} | Skipped (Dataset Build Fail)")
            continue
            
        scaler = StandardScaler()
        Xtr_x = scaler.fit_transform(Xtr_x)
        Xte_x = scaler.transform(Xte_x)
        
        Xtr = np.concatenate([Xtr_x, Xtr_m], axis=1)
        Xte = np.concatenate([Xte_x, Xte_m], axis=1)
        
        clf = MLPClassifier(hidden_layer_sizes=(128,), random_state=0, max_iter=1000, learning_rate_init=1e-3, alpha=1e-4)
        clf.fit(Xtr, ytr)
        
        probs = clf.predict_proba(Xte)[:, 1]
        
        best_idx = np.argmax(probs)
        predicted_movie = final_test[best_idx]
        price = probs[best_idx]
        
        actual_winner = best_picture_winners.get(year, "Unknown")
        
        # Check if Actual Winner is even in the test set!
        if actual_winner not in final_test:
            # If the actual winner isn't in our data, we can't possibly win, 
            # or the backtest is invalid for this year.
            # But let's assume it's a LOSS since we bet on something else.
            # Mark it in the output.
            actual_winner_disp = f"{actual_winner} (Not in Data)"
            did_win = False
        else:
            actual_winner_disp = actual_winner
            did_win = (predicted_movie == actual_winner)
        
        pnl = (1.0 - price) if did_win else (-price)
        cumulative_pnl += pnl
        
        res_str = "WIN" if did_win else "LOSS"
        
        print(f"{year:<6} | {train_start}-{train_end:<4} | {predicted_movie[:33]:<35} | ${price:.4f}  | {actual_winner_disp[:28]:<30} | {res_str:<6} | ${pnl:+.4f}")

    print("-" * 120)
    print(f"Total PnL (betting 1 unit on top pick each year): ${cumulative_pnl:.4f}")
    print(f"Average Profit per Year: ${cumulative_pnl/len(years_to_test):.4f}")

if __name__ == "__main__":
    run_backtest()