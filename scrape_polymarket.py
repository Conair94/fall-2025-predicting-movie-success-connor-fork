import requests
import pandas as pd
import json
import os
import argparse
from datetime import datetime

# --- Configuration ---
POLYMARKET_API_URL = "https://gamma-api.polymarket.com/events"
OUTPUT_FILE = "data/raw/market_odds_polymarket.csv"

def fetch_best_picture_market(year_str):
    """
    Fetches the 'Best Picture' market for a given Oscars year.
    Note: Oscars 2026 implies the ceremony in early 2026.
    Search query usually needs to be broad like "Oscar" or "Best Picture".
    """
    params = {
        "limit": 20,
        "active": "true",
        "closed": "false",
        "order": "volume24hr",
        "ascending": "false",
        "q": f"Best Picture {year_str}" # e.g. "Best Picture 2025" or "2026"
    }
    
    print(f"Searching Polymarket for: {params['q']}...")
    try:
        response = requests.get(POLYMARKET_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return None

    if not data:
        print("No events found.")
        return None

    # Filter for the specific market
    # We want the main "Winner" market.
    markets_data = []
    
    for event in data:
        # Check title relevance
        title = event.get('title', '')
        if 'Best Picture' not in title:
            continue
            
        print(f"Found Event: {title}")
        
        # Each event has 'markets'
        markets = event.get('markets', [])
        for m in markets:
            # We want the market outcomes
            # Polymarket outcome prices are usually in 'outcomePrices' (json string) or separate fields
            # The API structure varies, usually it returns a list of markets.
            # Let's extract outcomes.
            
            # Gamma API structure:
            # Market has 'outcomes' (["Movie A", "Movie B"]) and 'outcomePrices' (["0.10", "0.90"])
            outcomes = json.loads(m.get('outcomes', '[]'))
            prices = json.loads(m.get('outcomePrices', '[]'))
            
            if len(outcomes) != len(prices):
                print(f"Warning: Outcome/Price mismatch for {m.get('question')}")
                continue
                
            for outcome, price in zip(outcomes, prices):
                markets_data.append({
                    'market_id': m.get('id'),
                    'event_title': title,
                    'movie': outcome,
                    'price': float(price),
                    'volume': float(m.get('volume', 0)),
                    'timestamp': datetime.now().isoformat()
                })
                
    return pd.DataFrame(markets_data)

def main():
    parser = argparse.ArgumentParser(description="Scrape Polymarket for Oscar Odds")
    parser.add_argument("year", type=str, help="Oscar Ceremony Year (e.g. 2025 or 2026)")
    args = parser.parse_args()
    
    df = fetch_best_picture_market(args.year)
    
    if df is not None and not df.empty:
        # Clean Movie Names
        # Remove " (2024)" etc if present? Polymarket names are usually just the movie title.
        # But sometimes "Oppenheimer"
        
        # Save
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved {len(df)} market odds to {OUTPUT_FILE}")
        print(df.sort_values('price', ascending=False).head(10))
    else:
        print("No data found. Check your search query or Polymarket availability.")

if __name__ == "__main__":
    main()
