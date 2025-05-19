import json
import boto3
import requests
import logging
import os
import pandas as pd
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Comma-separated market keys for NBA/NCAAB/WNBA player props
PLAYER_PROP_MARKETS = (
    "player_points,player_points_q1,player_rebounds,player_rebounds_q1,"
    "player_assists,player_assists_q1,player_threes,player_blocks,player_steals,"
    "player_blocks_steals,player_turnovers,player_points_rebounds_assists,"
    "player_points_rebounds,player_points_assists,player_rebounds_assists,"
    "player_field_goals,player_frees_made,player_frees_attempts,player_first_basket,"
    "player_double_double,player_triple_double,player_method_of_first_basket"
)

class PlayerPropsAggregator:
    def __init__(self, api_key: str, sport_keys: List[str], regions: str = "us,eu,us_ex", odds_format: str = "american"):
        self.api_key = api_key
        self.sport_keys = sport_keys  # e.g. ["basketball_nba", "basketball_ncaab", "wnba"]
        self.regions = regions
        self.odds_format = odds_format
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.environ['S3_BUCKET_NAME']
    
    def fetch_events(self, sport_key: str) -> List[Dict[str, Any]]:
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events"
        params = {
            "apiKey": self.api_key,
            "regions": self.regions,
            "dateFormat": "iso"
        }
        try:
            logger.info(f"Fetching events for {sport_key}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            events = response.json()
            logger.info(f"Found {len(events)} events for {sport_key}")
            return events
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching events for {sport_key}: {e}")
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
            logger.info(f"Fetching odds for event {event_id} in {sport_key}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            odds = response.json()
            return odds
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching event odds for event {event_id}: {e}")
            return {}
    
    def create_prop_key(self, game_id: str, market: str, player_name: str) -> str:
        return f"{game_id}:{market}:{player_name}".lower()
    
    def get_bet_id(self, row):
        """
        Generate a unique identifier for a bet using:
          - game_id, market, player_name, and any columns ending in
            '_line', '_over_price', or '_under_price'.
        """
        base_id = f"{row.get('game_id', '')}_{row.get('market', '')}_{row.get('player_name', '')}"
        extra_parts = []
        for col in row.index:
            if col.endswith('_line') or col.endswith('_over_price') or col.endswith('_under_price'):
                value = row[col]
                if pd.notnull(value):
                    extra_parts.append(f"{col}:{value}")
        extra_parts.sort()
        full_id = base_id + ("_" + "_".join(extra_parts) if extra_parts else "")
        return full_id
    
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
            #Skip bettrivers
            if bookmaker_key == "betrivers":
                continue
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                # Only process markets that match one of our player prop market keys
                if market_key not in PLAYER_PROP_MARKETS.split(","):
                    continue
                for outcome in market.get("outcomes", []):
                    # For player props, expect a prop line and a player name
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
                            "player_team": None,       # Not provided by API
                            "player_position": None,   # Not provided by API
                            "timestamp": datetime.now().isoformat()
                        }
                    # Save line and price for "over" or "under"
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
    
    def update_active_bets(self, new_df):
        """
        Update the active bets and track bet durations in S3
        """
        current_run_time = datetime.now()
        
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], errors='coerce')
        
        if 'bet_id' not in new_df.columns:
            new_df['bet_id'] = new_df.apply(self.get_bet_id, axis=1)
        
        # Try to read existing active bets from S3
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key='active/player_props_active.csv'
            )
            active_df = pd.read_csv(response['Body'])
            active_df['timestamp'] = pd.to_datetime(active_df['timestamp'], errors='coerce')
        except Exception as e:
            logger.info(f"No active bets file found or error reading file: {e}")
            active_df = pd.DataFrame(columns=new_df.columns.tolist() + ['duration'])
        
        if 'duration' not in active_df.columns:
            active_df['duration'] = 0
        
        active_dict = {row['bet_id']: row for _, row in active_df.iterrows()}
        
        updated_active = []
        expired_bets = []
        
        for _, new_row in new_df.iterrows():
            bet_id = new_row['bet_id']
            if bet_id in active_dict:
                active_start = pd.to_datetime(active_dict[bet_id]['timestamp'])
                new_duration = (current_run_time - active_start).total_seconds()
                new_row = new_row.copy()  # Create a copy to avoid SettingWithCopyWarning
                new_row['duration'] = new_duration
                updated_active.append(new_row)
                del active_dict[bet_id]
            else:
                new_row = new_row.copy()  # Create a copy to avoid SettingWithCopyWarning
                new_row['timestamp'] = current_run_time
                new_row['duration'] = 0
                updated_active.append(new_row)
        
        for bet_id, expired_row in active_dict.items():
            # Compute final duration using the current run time
            start_time = pd.to_datetime(expired_row['timestamp'])
            expired_row['duration'] = (current_run_time - start_time).total_seconds()
            expired_bets.append(expired_row)
        
        # Upload updated active bets to S3
        updated_active_df = pd.DataFrame(updated_active)
        
        # Ensure bet_id is in the DataFrame
        if 'bet_id' not in updated_active_df.columns:
            updated_active_df['bet_id'] = updated_active_df.apply(self.get_bet_id, axis=1)
            
        csv_buffer = StringIO()
        updated_active_df.to_csv(csv_buffer, index=False)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key='raw/player_props_active.csv',
            Body=csv_buffer.getvalue()
        )
        
        # Handle expired bets - add to history
        if expired_bets:
            expired_df = pd.DataFrame(expired_bets)
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key='history/player_props_history.csv'
                )
                history_df = pd.read_csv(response['Body'])
                history_df = pd.concat([history_df, expired_df], ignore_index=True)
            except Exception as e:
                logger.info(f"No history file found or error reading file: {e}")
                history_df = expired_df
            
            history_buffer = StringIO()
            history_df.to_csv(history_buffer, index=False)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key='history/player_props_history.csv',
                Body=history_buffer.getvalue()
            )
        
        logger.info(
            f"Active bets updated: {len(updated_active_df)} bets active; {len(expired_bets)} bets expired."
        )
        
        return updated_active_df, expired_bets
    
    def run(self) -> pd.DataFrame:
        all_props = {}
        for sport_key in self.sport_keys:
            events = self.fetch_events(sport_key)
            logger.info(f"Processing {len(events)} events for {sport_key}")
            for event in events:
                event_id = event.get("id")
                if not event_id:
                    continue
                odds = self.fetch_event_odds(sport_key, event_id)
                props = self.extract_player_props_from_event(odds, sport_key)
                all_props.update(props)
        
        if not all_props:
            logger.error("No player prop information found across all events.")
            return pd.DataFrame()
        
        # Convert props to DataFrame
        new_df = pd.DataFrame(list(all_props.values()))
        
        # Add bet_id to the dataframe
        new_df['bet_id'] = new_df.apply(self.get_bet_id, axis=1)
        
        # Update active bets and get the updated dataframe
        updated_active_df, _ = self.update_active_bets(new_df)
        
        # Get the current UTC timestamp
        now_utc = datetime.now(timezone.utc)

        # Filter out rows where game_start is 24 hours or more in the past
        if 'game_start' in updated_active_df.columns:
            updated_active_df['game_start'] = pd.to_datetime(updated_active_df['game_start'], errors='coerce', utc=True)
            updated_active_df = updated_active_df[updated_active_df['game_start'] >= (now_utc - pd.Timedelta(days=1))]
        
        # Save the raw data for reference
        csv_buffer = StringIO()
        updated_active_df.to_csv(csv_buffer, index=False)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key='raw/player_props.csv',
            Body=csv_buffer.getvalue()
        )
        
        return updated_active_df

def lambda_handler(event, context):
    try:
        # Get API key from env var
        api_key = os.environ['API_KEY']
        s3_bucket = os.environ['S3_BUCKET_NAME']
        
        # Set sport keys
        sport_keys = ["basketball_nba", "basketball_ncaab"]
        
        # Create aggregator
        aggregator = PlayerPropsAggregator(
            api_key=api_key,
            sport_keys=sport_keys
        )
        
        # Run the aggregator to get DataFrame
        df = aggregator.run()
        
        if not df.empty:
            logger.info(f"Successfully processed player props data and updated S3 files in bucket {s3_bucket}")
            
            # Return success
            return {
                'statusCode': 200,
                'body': json.dumps('Successfully fetched player props data and updated S3 files')
            }
        else:
            logger.error("No data was fetched")
            return {
                'statusCode': 400,
                'body': json.dumps('No data was fetched')
            }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }