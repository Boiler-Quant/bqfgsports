import pandas as pd

# Load the dataset
file_path = '/Users/sebastianbohrt/Documents/GitHub/bqfgsports/player_props.csv' 
player_props_data = pd.read_csv(file_path)

# Define a function to calculate implied probability
def implied_probability(odds):
    if odds > 0:
        return 100 / (odds + 100)
    elif odds < 0:
        return -odds / (-odds + 100)
    return None

# Define a function to calculate the vig
def calculate_vig(over_odds, under_odds):
    prob_over = implied_probability(over_odds)
    prob_under = implied_probability(under_odds)
    if prob_over is not None and prob_under is not None:
        total_prob = prob_over + prob_under
        return total_prob - 1
    return None

# Define a function to calculate no-vig probabilities
def calculate_no_vig_probabilities(over_odds, under_odds):
    prob_over = implied_probability(over_odds)
    prob_under = implied_probability(under_odds)
    total_prob = prob_over + prob_under
    
    no_vig_prob_over = prob_over / total_prob
    no_vig_prob_under = prob_under / total_prob
    
    return no_vig_prob_over, no_vig_prob_under

# Define a function to convert probabilities to American odds
def probability_to_american_odds(prob):
    if prob > 0.5:
        return -100 * (prob / (1 - prob))
    else:
        return 100 * ((1 - prob) / prob)

# List of sportsbooks
sportsbooks = ['betrivers', 'bovada', 'espn_bet', 'fliff', 'pinnacle', 'thescore_bet']
over_columns = [f"{book}_over_price" for book in sportsbooks]
under_columns = [f"{book}_under_price" for book in sportsbooks]
line_columns = [f"{book}_line" for book in sportsbooks]

# Initialize new columns
player_props_data['lowest_vig'] = None
player_props_data['vig_sportsbook'] = None
player_props_data['fair_odds_over'] = None
player_props_data['fair_odds_under'] = None
player_props_data['best_odds_over'] = None
player_props_data['best_odds_under'] = None
player_props_data['best_odds_over_sportsbook'] = None
player_props_data['best_odds_under_sportsbook'] = None
player_props_data['best_bet_line_over'] = None
player_props_data['best_bet_line_under'] = None

# Perform calculations for each row
for index, row in player_props_data.iterrows():
    sportsbook_vigs = {}
    best_over = None
    best_under = None
    best_over_sportsbook = None
    best_under_sportsbook = None
    best_bet_line_over = None
    best_bet_line_under = None
    
    # Calculate vig for each sportsbook and find the best payout odds
    for book, over_col, under_col, line_col in zip(sportsbooks, over_columns, under_columns, line_columns):
        over_odds = row.get(over_col)
        under_odds = row.get(under_col)
        line = row.get(line_col)
        
        if pd.notnull(over_odds) and pd.notnull(under_odds):
            # Calculate vig
            vig = calculate_vig(over_odds, under_odds)
            if vig is not None:
                sportsbook_vigs[book] = vig
            
            # Find the best odds for over
            if pd.notnull(over_odds) and (best_over is None or over_odds > best_over):
                best_over = over_odds
                best_over_sportsbook = book
                best_bet_line_over = line
            
            # Find the best odds for under
            if pd.notnull(under_odds) and (best_under is None or under_odds > best_under):
                best_under = under_odds
                best_under_sportsbook = book
                best_bet_line_under = line
    
    # Find the sportsbook with the lowest vig
    if sportsbook_vigs:
        lowest_vig_sportsbook = min(sportsbook_vigs, key=sportsbook_vigs.get)
        player_props_data.loc[index, 'lowest_vig'] = sportsbook_vigs[lowest_vig_sportsbook]
        player_props_data.loc[index, 'vig_sportsbook'] = lowest_vig_sportsbook
        
        # Get the odds for the sportsbook with the lowest vig
        over_odds = row[f"{lowest_vig_sportsbook}_over_price"]
        under_odds = row[f"{lowest_vig_sportsbook}_under_price"]
        
        # Calculate no-vig probabilities and fair odds
        no_vig_prob_over, no_vig_prob_under = calculate_no_vig_probabilities(over_odds, under_odds)
        fair_odds_over = probability_to_american_odds(no_vig_prob_over)
        fair_odds_under = probability_to_american_odds(no_vig_prob_under)
        
        player_props_data.loc[index, 'fair_odds_over'] = fair_odds_over
        player_props_data.loc[index, 'fair_odds_under'] = fair_odds_under
    
    # Add the best odds and associated sportsbooks/lines
    player_props_data.loc[index, 'best_odds_over'] = best_over
    player_props_data.loc[index, 'best_odds_under'] = best_under
    player_props_data.loc[index, 'best_odds_over_sportsbook'] = best_over_sportsbook
    player_props_data.loc[index, 'best_odds_under_sportsbook'] = best_under_sportsbook
    player_props_data.loc[index, 'best_bet_line_over'] = best_bet_line_over
    player_props_data.loc[index, 'best_bet_line_under'] = best_bet_line_under

# Ensure columns are numeric for processing
columns_to_round = ['lowest_vig', 'fair_odds_over', 'fair_odds_under', 'best_odds_over', 'best_odds_under']

for col in columns_to_round:
    if col in player_props_data.columns:
        player_props_data[col] = pd.to_numeric(player_props_data[col], errors='coerce')

# Round numeric columns to 2 decimal places
player_props_data[columns_to_round] = player_props_data[columns_to_round].round(2)

# Express `lowest_vig` as a percentage
if 'lowest_vig' in player_props_data.columns:
    player_props_data['lowest_vig'] = (player_props_data['lowest_vig'] * 100).round(2).astype(str) + '%'

# Save the dataset with all calculated columns to a single file
player_props_data.to_csv('combined_data_with_best_odds_and_lines.csv', index=False)

print("Dataset with fair odds, vig, and best odds saved to 'combined_data_with_best_odds_and_lines.csv'")
