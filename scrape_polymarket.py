import requests
import pandas as pd
import json
import os
import argparse
from datetime import datetime

# --- Configuration ---
POLYMARKET_API_URL = "https://gamma-api.polymarket.com/events"
OUTPUT_FILE = "data/raw/market_odds_polymarket.csv"

def fetch_market_by_slug(slug):
    """
    Fetches a specific event by its URL slug.
    """
    params = {
        "slug": slug
    }
    
    print(f"Direct lookup for Polymarket event slug: '{slug}'...")
    try:
        response = requests.get(POLYMARKET_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"API Error: {e}")
        return None

    if not data:
        print(f"No event found for slug: {slug}")
        return None

    # The API might return a list [event] or a single event dict
    if isinstance(data, list):
        if not data:
            return None
        event = data[0]
    else:
        event = data

    print(f"Found Event: {event.get('title')}")
    
    markets_data = []
    markets = event.get('markets', [])
    
    for m in markets:
        # Determine the Movie Name
        # In event groups, 'groupItemTitle' usually holds the specific outcome name (the movie)
        movie_name = m.get('groupItemTitle')
        
        # Fallback: Parse from question (e.g. "Will Anora win Best Picture?")
        if not movie_name:
            question = m.get('question', '')
            if 'Will ' in question and ' win' in question:
                movie_name = question.split('Will ')[1].split(' win')[0]
            else:
                movie_name = question # Last resort
        
        # Extract outcomes and prices
        outcomes = json.loads(m.get('outcomes', '[]'))
        prices = json.loads(m.get('outcomePrices', '[]'))
        
        if len(outcomes) != len(prices):
            # Try to get price from the individual market fields if outcomePrices is empty
            # Gamma API sometimes uses 'lastTradePrice' or 'bestBid'/'bestAsk'
            continue
            
        for outcome, price in zip(outcomes, prices):
            # We primarily want the "Yes" price (the cost to bet on the movie winning)
            if outcome.lower() == 'yes':
                markets_data.append({
                    'market_id': m.get('id'),
                    'event_title': event.get('title'),
                    'movie': movie_name,
                    'price': float(price),
                    'volume': float(m.get('volume', 0)),
                    'timestamp': datetime.now().isoformat()
                })
            
    return pd.DataFrame(markets_data)

def main():
    parser = argparse.ArgumentParser(description="Scrape Polymarket for Oscar Odds")
    parser.add_argument("year", type=str, help="Oscar Ceremony Year (e.g. 2026)")
    args = parser.parse_args()
    
    # Construct the slug based on the user's provided URL structure
    # URL: https://polymarket.com/event/oscars-2026-best-picture-winner
    target_slug = f"oscars-{args.year}-best-picture-winner"
    
    df = fetch_market_by_slug(target_slug)
    
    if df is not None and not df.empty:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved {len(df)} market odds to {OUTPUT_FILE}")
        print(df.sort_values('price', ascending=False).head(10))
    else:
        print("Failed to fetch market data.")

if __name__ == "__main__":
    main()