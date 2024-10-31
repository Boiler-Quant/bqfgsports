from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import concurrent.futures
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

NBA_PLAYERS = [
    "stephen-curry", "lebron-james", "kevin-durant", "luka-doncic",
    "giannis-antetokounmpo", "joel-embiid", "nikola-jokic", "jayson-tatum",
    "damian-lillard", "devin-booker", "trae-young", "donovan-mitchell",
    "ja-morant", "anthony-edwards", "shai-gilgeous-alexander"
]

SPORTSBOOK_MAP = {
    3: 'BetMGM',
    4: 'DraftKings',
    5: 'FanDuel',
    7: 'Caesars',
    8: 'ESPN_Bet'
}

def setup_driver():
    """Configure and return a Chrome WebDriver instance with optimized settings"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-logging")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("--disable-images")
    chrome_options.page_load_strategy = 'eager'
    
    return webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )

def generate_player_url(player_name):
    """Generate the URL for a player's props page"""
    return f"https://www.bettingpros.com/nba/odds/player-props/{player_name}/"

def extract_sportsbook_odds(prop, sportsbook_indices):
    """Extract odds for specific sportsbooks with error handling"""
    try:
        odds_cells = prop.find_all('div', class_='flex odds-offer__item')
        
        # Get best odds
        best_odds_container = prop.find('div', class_='odds-offer__item odds-offer__item--best-odds')
        best_over = best_under = None
        if best_odds_container:
            best_odds_buttons = best_odds_container.find_all('button', class_='odds-cell odds-cell--best')
            if len(best_odds_buttons) > 1:
                best_over = best_odds_buttons[0].get_text(strip=True)
                best_under = best_odds_buttons[1].get_text(strip=True)

        # Initialize sportsbook odds dictionary with None values
        sportsbook_odds = {book: {'over': None, 'under': None} for book in SPORTSBOOK_MAP.values()}

        # Extract odds for each sportsbook
        for idx in sportsbook_indices:
            if idx >= len(odds_cells):
                continue
                
            sportsbook_name = SPORTSBOOK_MAP.get(idx)
            if not sportsbook_name:
                continue
                
            current_cell = odds_cells[idx]
            buttons = current_cell.find_all(
                ['button', 'div'],
                class_=['odds-cell odds-cell--default', 'odds-cell odds-cell--best']
            )
            
            if len(buttons) >= 2:
                over_text = buttons[0].get_text(strip=True)
                under_text = buttons[1].get_text(strip=True)
                
                if over_text != 'NL' and under_text != 'NL':
                    sportsbook_odds[sportsbook_name]['over'] = over_text
                    sportsbook_odds[sportsbook_name]['under'] = under_text

        return best_over, best_under, sportsbook_odds
    except Exception as e:
        logging.error(f"Error extracting sportsbook odds: {e}")
        return None, None, {}

def wait_for_props_load(driver):
    """Wait for props to load with explicit wait conditions"""
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "odds-offer__item"))
        )
        return True
    except Exception:
        return False

def scrape_props(player):
    """Scrape props for a single player"""
    driver = None
    try:
        driver = setup_driver()
        url = generate_player_url(player)
        driver.get(url)
        
        if not wait_for_props_load(driver):
            logging.warning(f"Timeout waiting for props to load for {player}")
            return pd.DataFrame()

        # Execute a single smooth scroll to trigger dynamic loading
        driver.execute_script(
            "window.scrollTo({top: document.body.scrollHeight, behavior: 'smooth'});"
        )
        time.sleep(2)  # Brief wait for dynamic content

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        player_name = player.replace('-', ' ').title()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        props_data = []
        prop_names = soup.find_all('div', class_='odds-offer__item odds-offer__item--first')
        prop_containers = soup.find_all('div', class_='flex odds-offer')

        for i, prop in enumerate(prop_containers):
            if i >= len(prop_names):
                break
                
            prop_name = prop_names[i].get_text(strip=True)
            best_over, best_under, sportsbook_odds = extract_sportsbook_odds(
                prop, list(SPORTSBOOK_MAP.keys())
            )

            prop_data = {
                'Timestamp': timestamp,
                'Player Name': player_name,
                'Player Prop': f"{player_name} - {prop_name}",
                'Best Over': best_over,
                'Best Under': best_under
            }

            # Add sportsbook-specific odds
            for book, odds in sportsbook_odds.items():
                prop_data[f'{book} Over'] = odds['over']
                prop_data[f'{book} Under'] = odds['under']

            props_data.append(prop_data)

        return pd.DataFrame(props_data)

    except Exception as e:
        logging.error(f"Error scraping props for {player}: {e}")
        return pd.DataFrame()
    finally:
        if driver:
            driver.quit()

def save_data(all_props_data, csv_path):
    """Save the scraped data to CSV with error handling"""
    try:
        new_data_df = pd.concat(all_props_data, ignore_index=True)
        
        if os.path.exists(csv_path):
            existing_data_df = pd.read_csv(csv_path)
            updated_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
        else:
            updated_df = new_data_df

        updated_df.to_csv(csv_path, index=False)
        logging.info(f"Successfully saved data for {len(all_props_data)} players")
    except Exception as e:
        logging.error(f"Error saving data: {e}")

def main(csv_path, max_workers=8):
    """Main execution function with parallel processing"""
    start_time = time.time()
    all_props_data = []
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_player = {
                executor.submit(scrape_props, player): player 
                for player in NBA_PLAYERS
            }
            
            for future in concurrent.futures.as_completed(future_to_player):
                player = future_to_player[future]
                try:
                    df = future.result()
                    if not df.empty:
                        all_props_data.append(df)
                        logging.info(f"Successfully scraped {len(df)} props for {player}")
                    else:
                        logging.warning(f"No props found for {player}")
                except Exception as e:
                    logging.error(f"Error processing {player}: {e}")

        if all_props_data:
            save_data(all_props_data, csv_path)
        else:
            logging.warning("No data was scraped for any player")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
    
    finally:
        execution_time = time.time() - start_time
        logging.info(f"Total execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    main("player_props.csv")