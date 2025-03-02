import pandas as pd
import numpy as np
from collections import Counter

# Load the dataset
file_path = '/Users/jamieborst/Downloads/player_props (1).csv' 
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
        return (total_prob - 1) * 100
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
player_props_data['vig_from_best_odds'] = None
player_props_data['fair_odds_over'] = None
player_props_data['fair_odds_under'] = None
player_props_data['best_odds_over'] = None
player_props_data['best_odds_under'] = None
player_props_data['best_odds_over_sportsbook'] = None
player_props_data['best_odds_under_sportsbook'] = None
player_props_data['best_odds_over_line'] = None
player_props_data['best_odds_under_line'] = None
player_props_data['fair_line'] = None
player_props_data['fair_line_method'] = None

# Perform calculations for each row
for index, row in player_props_data.iterrows():
    # First, determine the fair line to use (median or most common)
    available_lines = []
    for line_col in line_columns:
        line = row.get(line_col)
        if pd.notnull(line):
            available_lines.append(line)
    
    if available_lines:
        # Calculate both median and most common line
        median_line = np.median(available_lines)
        most_common_line = Counter(available_lines).most_common(1)[0][0]
        
        # Choose which fair line method to use (for this implementation, we'll use most common if it exists, else median)
        if len(Counter(available_lines)) > 1 and Counter(available_lines).most_common(1)[0][1] > 1:
            fair_line = most_common_line
            fair_line_method = "most_common"
        else:
            fair_line = median_line
            fair_line_method = "median"
        
        player_props_data.loc[index, 'fair_line'] = fair_line
        player_props_data.loc[index, 'fair_line_method'] = fair_line_method
    else:
        continue  # Skip rows with no available lines
    
    # Group sportsbooks by line for selecting best odds
    over_odds_by_line = {}
    under_odds_by_line = {}
    
    for book, over_col, under_col, line_col in zip(sportsbooks, over_columns, under_columns, line_columns):
        over_odds = row.get(over_col)
        under_odds = row.get(under_col)
        line = row.get(line_col)
        
        if pd.notnull(line) and pd.notnull(over_odds):
            if line not in over_odds_by_line:
                over_odds_by_line[line] = []
            over_odds_by_line[line].append((over_odds, book))
            
        if pd.notnull(line) and pd.notnull(under_odds):
            if line not in under_odds_by_line:
                under_odds_by_line[line] = []
            under_odds_by_line[line].append((under_odds, book))
    
    # Find best over odds (lowest line, then best payout)
    best_over = None
    best_over_sportsbook = None
    best_over_line = None
    
    if over_odds_by_line:
        # Sort lines for over (ascending)
        sorted_lines = sorted(over_odds_by_line.keys())
        for line in sorted_lines:
            # Get the best odds at this line
            best_at_line = max(over_odds_by_line[line], key=lambda x: x[0])
            if best_over is None or line < best_over_line or (line == best_over_line and best_at_line[0] > best_over):
                best_over = best_at_line[0]
                best_over_sportsbook = best_at_line[1]
                best_over_line = line
    
    # Find best under odds (highest line, then best payout)
    best_under = None
    best_under_sportsbook = None
    best_under_line = None
    
    if under_odds_by_line:
        # Sort lines for under (descending)
        sorted_lines = sorted(under_odds_by_line.keys(), reverse=True)
        for line in sorted_lines:
            # Get the best odds at this line
            best_at_line = max(under_odds_by_line[line], key=lambda x: x[0])
            if best_under is None or line > best_under_line or (line == best_under_line and best_at_line[0] > best_under):
                best_under = best_at_line[0]
                best_under_sportsbook = best_at_line[1]
                best_under_line = line
    
    # For calculating no-vig odds, find best over/under odds at the fair line
    best_over_at_fair_line = None
    best_under_at_fair_line = None
    
    if fair_line in over_odds_by_line:
        best_over_at_fair_line = max(over_odds_by_line[fair_line], key=lambda x: x[0])[0]
    
    if fair_line in under_odds_by_line:
        best_under_at_fair_line = max(under_odds_by_line[fair_line], key=lambda x: x[0])[0]
    
    # Calculate vig using the best odds at the fair line
    if best_over_at_fair_line is not None and best_under_at_fair_line is not None:
        vig = calculate_vig(best_over_at_fair_line, best_under_at_fair_line)
        no_vig_prob_over, no_vig_prob_under = calculate_no_vig_probabilities(best_over_at_fair_line, best_under_at_fair_line)
        
        # Calculate fair odds
        fair_odds_over = probability_to_american_odds(no_vig_prob_over)
        fair_odds_under = probability_to_american_odds(no_vig_prob_under)
        
        # Update the DataFrame with calculated values
        player_props_data.loc[index, 'vig_from_best_odds'] = vig
        player_props_data.loc[index, 'fair_odds_over'] = fair_odds_over
        player_props_data.loc[index, 'fair_odds_under'] = fair_odds_under
    
    # Update the best odds, sportsbooks, and lines
    player_props_data.loc[index, 'best_odds_over'] = best_over
    player_props_data.loc[index, 'best_odds_under'] = best_under
    player_props_data.loc[index, 'best_odds_over_sportsbook'] = best_over_sportsbook
    player_props_data.loc[index, 'best_odds_under_sportsbook'] = best_under_sportsbook
    player_props_data.loc[index, 'best_odds_over_line'] = best_over_line
    player_props_data.loc[index, 'best_odds_under_line'] = best_under_line

# Ensure columns are numeric for processing
columns_to_round = ['vig_from_best_odds', 'fair_odds_over', 'fair_odds_under', 'best_odds_over', 'best_odds_under', 'fair_line']

for col in columns_to_round:
    if col in player_props_data.columns:
        player_props_data[col] = pd.to_numeric(player_props_data[col], errors='coerce')

# Round numeric columns to 2 decimal places
player_props_data[columns_to_round] = player_props_data[columns_to_round].round(2)

# Express `vig_from_best_odds` as a percentage
if 'vig_from_best_odds' in player_props_data.columns:
    player_props_data['vig_from_best_odds'] = pd.to_numeric(player_props_data['vig_from_best_odds'], errors='coerce')

# Save the dataset with all calculated columns to a single file
player_props_data.to_csv('/Users/jamieborst/Downloads/enhanced_betting_analysis.csv', index=False)

print("Enhanced dataset with fair lines, best odds, and vig calculations saved to 'enhanced_betting_analysis.csv'")
