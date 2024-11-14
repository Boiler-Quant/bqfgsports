import requests
import json
import pandas as pd
from datetime import datetime
import logging
from typing import List, Dict, Any

class OddsAggregator:
    def __init__(self, api_key: str, base_url: str, debug_file: str = None):
        self.api_key = api_key
        self.base_url = base_url
        self.debug_file = debug_file
        
        self.sportsbooks = {
            'betrivers': 'betrivers_nba.json',
            'espn_bet': 'espn_bet_nba.json',
            'fliff': 'fliff_nba.json',
            'pinnacle': 'pinnacle_nba.json',
            'bovada': 'bovada_nba.json',
            'thescore_bet': 'thescore_bet_nba.json'
        }
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)

    def get_base_url_for_sportsbook(self, sportsbook: str) -> str:
        base = self.base_url.rsplit('/', 1)[0]
        return f"{base}/{self.sportsbooks[sportsbook]}"

    def fetch_odds_data(self, sportsbook: str) -> Dict[str, Any]:
        if self.debug_file:
            try:
                with open(self.debug_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error reading debug file: {e}")
                return {}
        
        try:
            url = f"{self.get_base_url_for_sportsbook(sportsbook)}?key={self.api_key}"
            self.logger.info(f"Fetching data from {sportsbook}")
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching {sportsbook} data: {e}")
            return {}

    def create_prop_key(self, game_id: str, market: str, player_name: str) -> str:
        return f"{game_id}:{market}:{player_name}".lower()

    def extract_player_props(self, data: Dict[str, Any], sportsbook: str) -> Dict[str, Dict]:
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
                                'timestamp': datetime.now().isoformat(),
                                f'{sportsbook}_line': points,
                                f'{sportsbook}_over_price': None,
                                f'{sportsbook}_under_price': None
                            }
                        
                        if selection == 'over':
                            props_dict[prop_key][f'{sportsbook}_over_price'] = price
                        elif selection == 'under':
                            props_dict[prop_key][f'{sportsbook}_under_price'] = price
                            
                        if props_dict[prop_key][f'{sportsbook}_line'] is None:
                            props_dict[prop_key][f'{sportsbook}_line'] = points
        
        return props_dict

    def save_to_csv(self, props: List[Dict[str, Any]], filename: str):
        if not props:
            return
            
        new_df = pd.DataFrame(props)
        
        base_columns = [
            'game_id', 'game_start', 'game_status', 'away_team', 'home_team',
            'market', 'player_name', 'player_team', 'player_position', 'timestamp'
        ]
        
        for sportsbook in sorted(self.sportsbooks.keys()):
            base_columns.extend([
                f'{sportsbook}_line',
                f'{sportsbook}_over_price',
                f'{sportsbook}_under_price'
            ])
        
        for col in base_columns:
            if col not in new_df.columns:
                new_df[col] = None
        
        new_df = new_df[base_columns]
        
        try:
            existing_df = pd.read_csv(filename)
            
            def create_identifier(row):
                return f"{row['game_id']}:{row['market']}:{row['player_name']}".lower()
            
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df['identifier'] = combined_df.apply(create_identifier, axis=1)
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            
            latest_df = (combined_df
                        .sort_values('timestamp', ascending=False)
                        .drop_duplicates('identifier')
                        .drop('identifier', axis=1)
                        .sort_values(['game_start', 'player_name', 'market']))
            
        except (FileNotFoundError, pd.errors.EmptyDataError):
            latest_df = new_df
        
        latest_df.to_csv(filename, index=False)
        self.logger.info(f"Updated {filename} - Total unique props: {len(latest_df)}")

    def run_once(self, output_file: str):
        try:
            all_props = {}
            
            for sportsbook in self.sportsbooks:
                data = self.fetch_odds_data(sportsbook)
                if data:
                    props = self.extract_player_props(data, sportsbook)
                    for key, prop in props.items():
                        if key in all_props:
                            all_props[key].update(prop)
                        else:
                            all_props[key] = prop
            
            merged_props = list(all_props.values())
            if merged_props:
                self.save_to_csv(merged_props, output_file)
                
        except Exception as e:
            self.logger.error(f"Error: {e}")

def main():
    API_KEY = "OBFMutcQkGXgddcPessefTL"
    BASE_URL = "https://data.oddsblaze.com/v1/odds/betrivers_nba.json"
    OUTPUT_FILE = "player_props.csv"
    DEBUG_FILE = ""  # Set to None for live mode
    
    aggregator = OddsAggregator(
        api_key=API_KEY,
        base_url=BASE_URL,
        debug_file=DEBUG_FILE
    )
    
    aggregator.run_once(output_file=OUTPUT_FILE)

if __name__ == "__main__":
    main()