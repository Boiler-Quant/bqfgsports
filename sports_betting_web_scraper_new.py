# Need to add logic to skip NL
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
from bs4 import BeautifulSoup
import pandas as pd
import os

# Setup Selenium with headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Navigate to the base page with player props
base_url = "https://www.bettingpros.com/nfl/odds/player-props/"
driver.get(base_url)

# Wait for the dynamic content to load
time.sleep(5)

# Initialize a list to store the player links
player_links = []

# Find all player prop links on the page
player_cards = driver.find_elements("css selector", ".card-body.player-card .flex a.player-card__link")
for card in player_cards[:15]:  # Limit to the first 15 players
    player_links.append(card.get_attribute('href'))

# Initialize a list to store the player props data
all_player_props = []

# Loop through each player link and scrape the data
for url in player_links:
    driver.get(url)
    time.sleep(5)  # Allow the page to load

    # Scroll the page to load additional content if necessary
    for _ in range(3):
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)
        driver.execute_script("window.scrollBy(0, -500);")
        time.sleep(1)
        driver.execute_script("window.scrollBy(0, 1000);")
        time.sleep(2)

    # Get the page source and parse with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    # Extract the player's name from the URL
    player_name = url.split('/')[-2].replace('-', ' ').title()

    # Extract player prop names
    prop_names = soup.find_all('div', class_='odds-offer__item odds-offer__item--first')
    prop_names_text = [prop.get_text(strip=True) for prop in prop_names]

    # Find all player prop containers for odds
    prop_containers = soup.find_all('div', class_='flex odds-offer')

    # Loop through each prop container and match with prop names
    for i, prop in enumerate(prop_containers):
        if i < len(prop_names_text):  # Ensure index is within bounds
            prop_name = prop_names_text[i]  # Get the corresponding prop name

            # Prepend the player's name to the prop name
            full_prop_name = f"{player_name} - {prop_name}"

            # Extract best odds
            best_odds_container = prop.find('div', class_='odds-offer__item odds-offer__item--best-odds')
            odds_buttons_best = best_odds_container.find_all('button', class_='odds-cell odds-cell--best')

            if len(odds_buttons_best) >= 2:  # Ensure both over and under are present
                over_odds_best_full = odds_buttons_best[0].get_text(strip=True)
                under_odds_best_full = odds_buttons_best[1].get_text(strip=True)

                # Split the over and under odds
                over_value_best, over_odds_best = over_odds_best_full.split('(')
                under_value_best, under_odds_best = under_odds_best_full.split('(')

                # Clean up the strings
                over_value_best = over_value_best.strip()
                over_odds_best = f'({over_odds_best.strip()[:-1]})'  # Remove trailing ')'
                under_value_best = under_value_best.strip()
                under_odds_best = f'({under_odds_best.strip()[:-1]})'  # Remove trailing ')'

            # Extract consensus odds
            consensus_odds_container = prop.find_all('div', class_='flex odds-offer__item')[-1]
            odds_buttons_consensus = consensus_odds_container.find_all('button', class_='odds-cell odds-cell--default')

            if len(odds_buttons_consensus) >= 2:  # Ensure both over and under are present
                over_odds_consensus_full = odds_buttons_consensus[0].get_text(strip=True)
                under_odds_consensus_full = odds_buttons_consensus[1].get_text(strip=True)

                # Split the over and under odds for consensus
                over_value_consensus, over_odds_consensus = over_odds_consensus_full.split('(')
                under_value_consensus, under_odds_consensus = under_odds_consensus_full.split('(')

                # Clean up the strings
                over_value_consensus = over_value_consensus.strip()
                over_odds_consensus = f'({over_odds_consensus.strip()[:-1]})'  # Remove trailing ')'
                under_value_consensus = under_value_consensus.strip()
                under_odds_consensus = f'({under_odds_consensus.strip()[:-1]})'  # Remove trailing ')'

                all_player_props.append({
                    'Player Prop': full_prop_name,
                    'Over Best Odds': over_value_best,
                    'Over Best Odds Value': over_odds_best,
                    'Under Best Odds': under_value_best,
                    'Under Best Odds Value': under_odds_best,
                    'Over Consensus Odds': over_value_consensus,
                    'Over Consensus Odds Value': over_odds_consensus,
                    'Under Consensus Odds': under_value_consensus,
                    'Under Consensus Odds Value': under_odds_consensus,
                })

# Convert the list to a DataFrame
new_data_df = pd.DataFrame(all_player_props)

# Replace 'EVEN' with '+100' in the relevant columns
new_data_df['Over Best Odds Value'] = new_data_df['Over Best Odds Value'].replace('(EVEN)', '(+100)')
new_data_df['Under Best Odds Value'] = new_data_df['Under Best Odds Value'].replace('(EVEN)', '(+100)')
new_data_df['Over Consensus Odds Value'] = new_data_df['Over Consensus Odds Value'].replace('(EVEN)', '(+100)')
new_data_df['Under Consensus Odds Value'] = new_data_df['Under Consensus Odds Value'].replace('(EVEN)', '(+100)')

# Define the path to the CSV file
csv_path = '/Users/jamieborst/Documents/Purdue Senior Year/BQFG/player_props.csv'

# Check if the CSV file exists
if os.path.exists(csv_path):
    # Load the existing data
    existing_data_df = pd.read_csv(csv_path)
    
    # Append the new data
    updated_df = pd.concat([existing_data_df, new_data_df], ignore_index=True)
else:
    # If the file does not exist, the new data is the updated data
    updated_df = new_data_df

# Save the updated DataFrame to CSV (overwrite the file)
updated_df.to_csv(csv_path, index=False)

# Close the Selenium browser
driver.quit()

print("Data has been successfully scraped, updated, and saved.")