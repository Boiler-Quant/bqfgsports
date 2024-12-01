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
def calculate_vig(row):
    prob_over = implied_probability(row['pinnacle_over_price'])
    prob_under = implied_probability(row['pinnacle_under_price'])
    total_prob = prob_over + prob_under
    return total_prob - 1

# Define a function to calculate no-vig probabilities
def calculate_no_vig_probabilities(row):
    prob_over = implied_probability(row['pinnacle_over_price'])
    prob_under = implied_probability(row['pinnacle_under_price'])
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

# Define a function to calculate fair odds
def add_fair_odds(row):
    no_vig_prob_over, no_vig_prob_under = calculate_no_vig_probabilities(row)
    fair_odds_over = probability_to_american_odds(no_vig_prob_over)
    fair_odds_under = probability_to_american_odds(no_vig_prob_under)
    return pd.Series([fair_odds_over, fair_odds_under, no_vig_prob_over, no_vig_prob_under], 
                     index=['fair_odds_over', 'fair_odds_under', 'no_vig_prob_over', 'no_vig_prob_under'])

# Initialize new columns with NaN
player_props_data['vig'] = None
player_props_data['fair_odds_over'] = None
player_props_data['fair_odds_under'] = None
player_props_data['no_vig_prob_over'] = None
player_props_data['no_vig_prob_under'] = None

# Perform calculations for rows with Pinnacle odds
for index, row in player_props_data.iterrows():
    if not pd.isnull(row['pinnacle_over_price']) and not pd.isnull(row['pinnacle_under_price']):
        # Calculate vig
        player_props_data.loc[index, 'vig'] = calculate_vig(row)
        
        # Calculate fair odds
        fair_odds = add_fair_odds(row)
        player_props_data.loc[index, ['fair_odds_over', 'fair_odds_under', 'no_vig_prob_over', 'no_vig_prob_under']] = fair_odds

# Define a list of sportsbooks for consensus odds calculation
sportsbooks = [
    'betrivers', 'bovada', 'espn_bet', 'fliff', 'pinnacle', 'thescore_bet'
]
over_columns = [f"{book}_over_price" for book in sportsbooks]
under_columns = [f"{book}_under_price" for book in sportsbooks]

# Define a function to calculate consensus odds
def calculate_consensus_odds(row, columns):
    odds = [row[col] for col in columns if not pd.isnull(row[col])]
    if len(odds) > 0:
        return sum(odds) / len(odds)
    else:
        return None  

# Add consensus odds for over and under
player_props_data['consensus_over_odds'] = player_props_data.apply(calculate_consensus_odds, axis=1, columns=over_columns)
player_props_data['consensus_under_odds'] = player_props_data.apply(calculate_consensus_odds, axis=1, columns=under_columns)

# Save the dataset with all calculated columns to a single file
player_props_data.to_csv('combined_data_with_fair_and_consensus_odds.csv', index=False)

print("Dataset with fair odds, vig, and consensus odds saved to 'combined_data_with_fair_and_consensus_odds.csv'")
