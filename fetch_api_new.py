import requests
import json
import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict, Any

# Comma-separated market keys for NBA/NCAAB/WNBA player props (primary and alternate)
PLAYER_PROP_MARKETS = (
    "player_points,player_points_q1,player_rebounds,player_rebounds_q1,"
    "player_assists,player_assists_q1,player_threes,player_blocks,player_steals,"
    "player_blocks_steals,player_turnovers,player_points_rebounds_assists,"
    "player_points_rebounds,player_points_assists,player_rebounds_assists,"
    "player_field_goals,player_frees_made,player_frees_attempts,player_first_basket,"
    "player_double_double,player_triple_double,player_method_of_first_basket,"
    "player_points_alternate,player_rebounds_alternate,player_assists_alternate,"
    "player_blocks_alternate,player_steals_alternate,player_turnovers_alternate,"
    "player_threes_alternate,player_points_assists_alternate,player_points_rebounds_alternate,"
    "player_rebounds_assists_alternate,player_points_rebounds_assists_alternate"
)

class PlayerPropsAggregator:
    def __init__(self, api_key: str, sport_keys: List[str], regions: str = "us", odds_format: str = "decimal", debug_file: str = None):
        self.api_key = api_key
        self.sport_keys = sport_keys  # e.g. ["basketball_nba", "basketball_ncaab", "wnba"]
        self.regions = regions
        self.odds_format = odds_format
        self.debug_file = debug_file
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    def fetch_events(self, sport_key: str) -> List[Dict[str, Any]]:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "dateFormat": "iso"
        }
        try:
            self.logger.info(f"Fetching events for {sport_key}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            events = response.json()
            self.logger.info(f"Found {len(events)} events for {sport_key}")
            return events
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching events for {sport_key}: {e}")
            return []
    
    def fetch_event_odds(self, sport_key: str, event_id: str) -> Dict[str, Any]:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "oddsFormat": self.odds_format,
            "dateFormat": "iso",
            "markets": PLAYER_PROP_MARKETS
        }
        try:
            self.logger.info(f"Fetching odds for event {event_id} in {sport_key}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            odds = response.json()
            return odds
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching event odds for event {event_id}: {e}")
            return {}
    
    def create_prop_key(self, game_id: str, market: str, player_name: str) -> str:
        return f"{game_id}:{market}:{player_name}".lower()
    
    def extract_player_props_from_event(self, event_odds: Dict[str, Any], sport_key: str) -> Dict[str, Dict]:
        props_dict = {}
        if not event_odds:
            return props_dict
        game_id = event_odds.get("id")
        game_start = event_odds.get("commence_time")
        home_team = event_odds.get("home_team")
        away_team = event_odds.get("away_team")
        for bookmaker in event_odds.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key")
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                # Only process markets that exactly match one of our player prop market keys
                if market_key not in PLAYER_PROP_MARKETS.split(","):
                    continue
                for outcome in market.get("outcomes", []):
                    # For player props, we expect a prop line ("point") and a player name in "description"
                    if outcome.get("point") is None or "description" not in outcome:
                        continue
                    selection = outcome.get("name", "").lower()  # "over" or "under"
                    player_name = outcome.get("description")
                    line = outcome.get("point")
                    price = outcome.get("price")
                    if not all([game_id, market_key, player_name]):
                        continue
                    prop_key = self.create_prop_key(game_id, market_key, player_name)
                    if prop_key not in props_dict:
                        props_dict[prop_key] = {
                            "sport": sport_key,
                            "game_id": game_id,
                            "game_start": game_start,
                            "away_team": away_team,
                            "home_team": home_team,
                            "market": market_key,
                            "player_name": player_name,
                            "player_team": None,       # Not provided by API response
                            "player_position": None,   # Not provided by API response
                            "timestamp": datetime.now().isoformat()
                        }
                    # Save the prop line and price for "over" or "under"
                    line_key = f"{bookmaker_key}_line"
                    if props_dict[prop_key].get(line_key) is None:
                        props_dict[prop_key][line_key] = line
                    if selection == "over":
                        over_price_key = f"{bookmaker_key}_over_price"
                        props_dict[prop_key][over_price_key] = price
                    elif selection == "under":
                        under_price_key = f"{bookmaker_key}_under_price"
                        props_dict[prop_key][under_price_key] = price
        return props_dict
    
    def run(self, output_file: str):
        all_props = {}
        for sport_key in self.sport_keys:
            events = self.fetch_events(sport_key)
            self.logger.info(f"Processing {len(events)} events for {sport_key}")
            for event in events:
                event_id = event.get("id")
                if not event_id:
                    continue
                odds = self.fetch_event_odds(sport_key, event_id)
                props = self.extract_player_props_from_event(odds, sport_key)
                all_props.update(props)
        if all_props:
            self.save_to_csv(all_props, output_file)
        else:
            self.logger.error("No player prop information found across all events.")
    
    def save_to_csv(self, props: Dict[str, Dict], filename: str):
        new_df = pd.DataFrame(list(props.values()))
        base_columns = [
            "sport", "game_id", "game_start", "away_team", "home_team",
            "market", "player_name", "player_team", "player_position", "timestamp"
        ]
        extra_columns = [col for col in new_df.columns if col not in base_columns]
        columns_order = base_columns + sorted(extra_columns)
        for col in columns_order:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df[columns_order]
        try:
            existing_df = pd.read_csv(filename)
            def create_identifier(row):
                return f"{row['game_id']}:{row['market']}:{row['player_name']}".lower()
            
            # Convert timestamp columns to datetime with proper format handling
            # For the existing_df
            if 'timestamp' in existing_df.columns:
                existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'], errors='coerce')
            
            # For the new_df
            if 'timestamp' in new_df.columns:
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], errors='coerce')
            
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df["identifier"] = combined_df.apply(create_identifier, axis=1)
            
            # No need to convert timestamp again, as we've already done it
            latest_df = (combined_df
                         .sort_values("timestamp", ascending=False)
                         .drop_duplicates("identifier")
                         .drop("identifier", axis=1)
                         .sort_values(["game_start", "player_name", "market"]))
        except (FileNotFoundError, pd.errors.EmptyDataError):
            # If file doesn't exist, just use new_df and convert its timestamp
            if 'timestamp' in new_df.columns:
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], errors='coerce')
            latest_df = new_df
            
        latest_df.to_csv(filename, index=False)
        self.logger.info(f"Updated {filename} - Total unique props: {len(latest_df)}")

def main():
    API_KEY = "e000504b835e1d765a889d7e60a52a2c"
    # You can adjust the sport keys as needed. Here we try NBA, NCAAB, and WNBA.
    sport_keys = ["basketball_nba", "basketball_ncaab"]
    OUTPUT_FILE = "player_props.csv"
    DEBUG_FILE = None  # Set this to a file path for debugging if needed
    
    aggregator = PlayerPropsAggregator(
        api_key=API_KEY,
        sport_keys=sport_keys,
        debug_file=DEBUG_FILE
    )
    aggregator.run(output_file=OUTPUT_FILE)

if __name__ == "__main__":
    main()
