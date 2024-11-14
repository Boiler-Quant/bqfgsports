import requests
import json
import time
import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict, Any

class OddsAggregator:
    def __init__(self, api_key: str, base_url: str, requests_per_minute: int = 10, debug_file: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.min_interval = 60 / requests_per_minute
        self.debug_file = debug_file
        self.last_request_time = 0
        
        # Define sportsbooks and their endpoints
        self.sportsbooks = {
            'betrivers': 'betrivers_nba.json',
            'espn_bet': 'espn_bet_nba.json',
            'fliff': 'fliff_nba.json',
            'pinnacle': 'pinnacle_nba.json'
        }
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def get_base_url_for_sportsbook(self, sportsbook: str) -> str:
        """Construct the full URL for a specific sportsbook"""
        base = self.base_url.rsplit('/', 1)[0]
        return f"{base}/{self.sportsbooks[sportsbook]}"

    def fetch_odds_data(self, sportsbook: str) -> Dict[str, Any]:
        """Fetch data from the API with rate limiting for a specific sportsbook"""
        if self.debug_file:
            try:
                with open(self.debug_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading debug file: {e}")
                return {}
        
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.min_interval:
            time.sleep(self.min_interval - time_since_last_request)
        
        try:
            url = f"{self.get_base_url_for_sportsbook(sportsbook)}?key={self.api_key}"
            self.logger.info(f"Fetching data from {sportsbook}")
            response = requests.get(url)
            response.raise_for_status()
            self.last_request_time = time.time()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching {sportsbook} data: {e}")
            return {}

    def create_prop_key(self, game_id: str, market: str, player_name: str) -> str:
        """Create a unique key for each prop bet to match across sportsbooks"""
        # Remove specific numbers from market to match different lines
        market_type = ''.join([c for c in market if not c.isdigit() and c != '.'])
        return f"{game_id}:{market_type}:{player_name}".lower()

    def extract_player_props(self, data: Dict[str, Any], sportsbook: str) -> Dict[str, Dict]:
        """Extract player prop data from the API response"""
        props_dict = {}
        
        games = data.get('games', [])
        for game in games:
            game_id = game.get('id')
            game_start = game.get('start')
            game_status = game.get('status')
            teams = game.get('teams', {})
            away_team = teams.get('away', {}).get('name')
            home_team = teams.get('home', {}).get('name')
            
            for sb in game.get('sportsbooks', []):
                for odd in sb.get('odds', []):
                    market = odd.get('market', '')
                    if 'player' in market.lower():
                        player = odd.get('players', [{}])[0] if odd.get('players') else {}
                        points = odd.get('points')
                        selection = odd.get('selection', '').lower()
                        player_name = player.get('name')
                        price = odd.get('price')
                        
                        if not all([game_id, market, player_name]):
                            continue
                            
                        prop_key = self.create_prop_key(game_id, market, player_name)
                        
                        if prop_key not in props_dict:
                            props_dict[prop_key] = {
                                'game_id': game_id,
                                'game_start': game_start,
                                'game_status': game_status,
                                'away_team': away_team,
                                'home_team': home_team,
                                'market': market,
                                'player_name': player_name,
                                'player_team': player.get('team', {}).get('name'),
                                'player_position': player.get('position'),
                                'timestamp': datetime.now().isoformat()
                            }
                        
                        # Add sportsbook-specific line and price
                        if selection == 'over':
                            props_dict[prop_key][f'{sportsbook}_line'] = points
                            props_dict[prop_key][f'{sportsbook}_over_price'] = price
                        elif selection == 'under':
                            props_dict[prop_key][f'{sportsbook}_under_price'] = price
        
        return props_dict

    def merge_sportsbook_data(self, all_props: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Merge data from all sportsbooks into a single list of records"""
        return list(all_props.values())

    def save_to_csv(self, props: List[Dict[str, Any]], filename: str):
        """Save the player props data to a CSV file"""
        if not props:
            return
            
        df = pd.DataFrame(props)
        
        # Sort columns to maintain consistent order
        columns = ['game_id', 'game_start', 'game_status', 'away_team', 'home_team', 
                  'market', 'player_name', 'player_team', 'player_position', 'timestamp']
        
        # Add sportsbook columns in a specific order
        for sportsbook in sorted(self.sportsbooks.keys()):
            columns.extend([
                f'{sportsbook}_line',
                f'{sportsbook}_over_price',
                f'{sportsbook}_under_price'
            ])
        
        # Reorder columns and fill missing values with None
        df = df.reindex(columns=columns)
        
        try:
            existing_df = pd.read_csv(filename)
            df = pd.concat([existing_df, df], ignore_index=True)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass
        
        df.to_csv(filename, index=False)
        self.logger.info(f"Updated {filename} with {len(props)} props")

    def run(self, output_file: str, iterations: int = None, delay: int = 60):
        """Main function to run the data collection process"""
        iteration = 0
        
        while iterations is None or iteration < iterations:
            try:
                if self.debug_file:
                    self.logger.info(f"Debug iteration {iteration + 1}")
                
                all_props = {}
                
                # Fetch data from each sportsbook
                for sportsbook in self.sportsbooks:
                    data = self.fetch_odds_data(sportsbook)
                    if data:
                        props = self.extract_player_props(data, sportsbook)
                        # Merge props with existing data
                        for key, prop in props.items():
                            if key in all_props:
                                all_props[key].update(prop)
                            else:
                                all_props[key] = prop
                
                # Convert merged data to list and save
                merged_props = self.merge_sportsbook_data(all_props)
                if merged_props:
                    self.save_to_csv(merged_props, output_file)
                
                iteration += 1
                
                if iterations is None and not self.debug_file:
                    time.sleep(delay)
                
            except Exception as e:
                self.logger.error(f"Error: {e}")
                time.sleep(delay)

def main():
    # Configuration
    API_KEY = "OBFMutcQkGXgddcPessefTL"
    BASE_URL = "https://data.oddsblaze.com/v1/odds/betrivers_nba.json"
    OUTPUT_FILE = "player_props.csv"
    DEBUG_FILE = ""  # Set to None for live API mode
    DELAY = 60  # Seconds between iterations
    
    # Initialize and run the aggregator
    aggregator = OddsAggregator(
        api_key=API_KEY,
        base_url=BASE_URL,
        requests_per_minute=10,
        debug_file=DEBUG_FILE
    )
    
    # Run with 1 iteration in debug mode, or indefinitely in live mode
    aggregator.run(
        output_file=OUTPUT_FILE,
        iterations=1 if DEBUG_FILE else None,
        delay=DELAY
    )

if __name__ == "__main__":
    main()