import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def fetch_nba_players():
    """
    Fetch current NBA players from basketball-reference.com
    Returns a list of player names in firstname-lastname format
    """
    try:
        # Use basketball-reference.com as it's regularly updated
        url = "https://www.basketball-reference.com/leagues/NBA_2024_per_game.html"
        
        # Add headers to mimic browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the page
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the player table
        player_table = soup.find('table', id='per_game_stats')
        
        if not player_table:
            raise ValueError("Could not find player statistics table")
        
        # Extract player names
        players = []
        rows = player_table.find_all('tr', class_=lambda x: x != 'thead')
        
        for row in rows:
            name_cell = row.find('td', {'data-stat': 'player'})
            if name_cell and name_cell.a:  # Only get players with links (active players)
                full_name = name_cell.a.text
                
                # Process the name
                # Handle special cases like "P.J. Tucker" or "R.J. Barrett"
                if '.' in full_name:
                    parts = full_name.split()
                    if len(parts) == 2:  # Like "P.J. Tucker"
                        firstname = parts[0].replace('.', '')
                        lastname = parts[1]
                    else:  # Like "R.J. Hampton"
                        firstname = parts[0].replace('.', '')
                        lastname = ' '.join(parts[2:])
                else:
                    # Split into first and last name
                    parts = full_name.split()
                    firstname = parts[0]
                    lastname = ' '.join(parts[1:])
                
                # Clean and format the name
                formatted_name = f"{firstname}-{lastname}".lower()
                formatted_name = formatted_name.replace(' ', '-')  # Handle multi-word last names
                formatted_name = formatted_name.replace("'", '')   # Remove apostrophes
                formatted_name = formatted_name.replace(".", '')   # Remove any remaining periods
                
                players.append({
                    'full_name': full_name,
                    'formatted_name': formatted_name
                })
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(players).drop_duplicates()
        
        # Sort by full name
        df = df.sort_values('full_name')
        
        # Save to CSV for reference
        df.to_csv('nba_players.csv', index=False)
        
        # Print some stats
        logging.info(f"Found {len(df)} NBA players")
        
        # Return the formatted names as a list
        return df['formatted_name'].tolist()
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching data: {e}")
        return []
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        return []

def display_sample_names(players, num_samples=10):
    """Display a sample of player names for verification"""
    if players:
        logging.info("\nSample of formatted names:")
        for player in sorted(players)[:num_samples]:
            print(player)

def main():
    start_time = time.time()
    
    # Fetch players
    players = fetch_nba_players()
    
    if players:
        # Display some statistics
        logging.info(f"\nTotal players found: {len(players)}")
        
        # Show sample of names
        display_sample_names(players)
        
        # Write the formatted list to a Python file
        with open('nba_players_list.py', 'w') as f:
            f.write("NBA_PLAYERS = [\n")
            for player in sorted(players):
                f.write(f"    \"{player}\",\n")
            f.write("]\n")
        
        logging.info(f"\nPlayer list has been saved to 'nba_players_list.py'")
        logging.info(f"Full player data has been saved to 'nba_players.csv'")
    else:
        logging.error("No players were found")
    
    execution_time = time.time() - start_time
    logging.info(f"\nExecution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main()