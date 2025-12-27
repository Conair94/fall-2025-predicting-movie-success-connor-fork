import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os

def load_data():
    """Loads raw review data and award labels."""
    try:
        # Load Reviews (Wide Format: User x Movies)
        # Assuming 'data/raw' is in the current directory
        df_winners = pd.read_csv('data/raw/user_award_reviews.csv')
        df_losers = pd.read_csv('data/raw/user_non_award_reviews.csv')
        
        # Load Dates/Labels
        # We need to know which movies are actually winners vs losers to calculate Alpha
        # The file names imply the split, but let's confirm with the date files if needed.
        # Ideally, we merge everything into a long format: [User, Movie, Rating, Is_Winner, Year]
        
        # Process Winners
        # Drop username to get list of movies, but keep username for melt
        winner_movies = [c for c in df_winners.columns if c != 'username' and c != 'user_id']
        df_w_long = df_winners.melt(id_vars=['username'], value_vars=winner_movies, var_name='movie', value_name='rating')
        df_w_long['is_winner'] = 1
        
        # Process Losers
        loser_movies = [c for c in df_losers.columns if c != 'username' and c != 'user_id']
        df_l_long = df_losers.melt(id_vars=['username'], value_vars=loser_movies, var_name='movie', value_name='rating')
        df_l_long['is_winner'] = 0
        
        # Combine
        full_data = pd.concat([df_w_long, df_l_long], axis=0)
        
        # Clean Ratings (remove 0s or NaNs which might imply no review)
        # Letterboxd ratings are 0.5 to 5.0. 0 often means "watched but not rated" or missing.
        # We should filter for valid ratings.
        full_data['rating'] = pd.to_numeric(full_data['rating'], errors='coerce')
        full_data = full_data[full_data['rating'] > 0]
        
        return full_data
        
    except FileNotFoundError:
        print("Error: source_data files not found. Please check directory structure.")
        return None

def calculate_academy_alpha(df, min_reviews=10):
    """
    Calculates 'Academy Alpha' (AUC) for each user.
    AUC = Probability that a randomly chosen Winner is rated higher than a randomly chosen Loser by this user.
    0.5 = Random guessing. 1.0 = Perfect prediction. <0.5 = Inverse prediction.
    """
    user_stats = []
    
    grouped = df.groupby('username')
    
    print(f"Analyzing {len(grouped)} users...")
    
    for user, group in grouped:
        n_reviews = len(group)
        if n_reviews < min_reviews:
            continue
            
        # Check if user has rated BOTH winners and losers (needed for AUC)
        if group['is_winner'].nunique() < 2:
            continue
            
        try:
            # AUC Score
            auc = roc_auc_score(group['is_winner'], group['rating'])
            
            # Mean Rating (for context)
            mean_rating = group['rating'].mean()
            
            user_stats.append({
                'username': user,
                'academy_alpha_auc': auc,
                'n_reviews': n_reviews,
                'mean_rating': mean_rating
            })
        except ValueError:
            continue
            
    return pd.DataFrame(user_stats)

def main():
    print("--- Loading Data ---")
    df = load_data()
    if df is None: return
    
    print(f"Loaded {len(df)} total reviews.")
    
    print("--- Calculating User Alpha (AUC) ---")
    # Calculate Alpha
    alpha_df = calculate_academy_alpha(df, min_reviews=20)
    
    if alpha_df.empty:
        print("No users met the criteria for Alpha calculation.")
        return
        
    # Rank
    alpha_df.sort_values('academy_alpha_auc', ascending=False, inplace=True)
    
    # Save
    output_path = 'user_academy_alpha_ranking.csv'
    alpha_df.to_csv(output_path, index=False)
    
    print(f"\n--- Top 20 'Super Forecaster' Users (by Academy Alpha) ---")
    print(alpha_df.head(20).to_string(index=False))
    
    print(f"\nSaved full ranking to {output_path}")
    print(f"Total Users Ranked: {len(alpha_df)}")

if __name__ == "__main__":
    main()
