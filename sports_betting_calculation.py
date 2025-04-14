import pandas as pd
import numpy as np

# --- Function Definitions --

# Function to calculate arbitrage opportunity
def calculate_arbitrage(over_odds, under_odds, bet_on_under):
    over_decimal = decimal_odds(over_odds)
    under_decimal = decimal_odds(under_odds)
    
    over_implied_prob = implied_probability(over_odds)
    under_implied_prob = implied_probability(under_odds)

    if over_implied_prob + under_implied_prob < 1:
        bet_on_over = (bet_on_under * under_decimal) / over_decimal
        return bet_on_over, bet_on_under
        
    return None, None

# Convert American odds to implied probability
def implied_probability(american_odds):
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return -american_odds / (-american_odds + 100)

# Convert American odds to payout multiplier
def decimal_odds(american_odds):
    if american_odds > 0:
        return (american_odds / 100) + 1
    elif american_odds < 0:
        return 100 / abs(american_odds) + 1

# Convert decimal odds to American odds
def decimal_to_american_odds(decimal_odds):
    if decimal_odds >= 2:
        return (decimal_odds - 1) * 100
    else:
        return -100 / (decimal_odds - 1)

# Convert pimplied probability to American odds
def prob_to_american_odds(prob):
    if prob > 0.5:
        return -100 * (prob / (1 - prob))
    else:
        return 100 * ((1 - prob) / prob)

# Calculate EV as a percentage
def expected_value(fair_odds, best_odds):
    if best_odds > fair_odds:
        win_prob = implied_probability(fair_odds)
        payout = 100 * decimal_odds(best_odds)
        edge = (win_prob * (payout - 100)) - ((1 - win_prob) * (100))
        return edge / 100

# Calculate Kelly Criterion as a percentage
def kelly_criterion(fair_odds, best_odds):
    win_prob = implied_probability(fair_odds)
    b = decimal_odds(best_odds) - 1
    q = 1 - win_prob
    return (win_prob * b - q) / b

# Calculate no-vig (fair) odds
def no_vig(over_odds, under_odds):
    over_implied_prob = implied_probability(over_odds)
    under_implied_prob = implied_probability(under_odds)
    sum = over_implied_prob + under_implied_prob 
    no_vig_over_odds = over_implied_prob / sum
    no_vig_under_odds = under_implied_prob / sum
    return prob_to_american_odds(no_vig_over_odds), prob_to_american_odds(no_vig_under_odds)

# Find the best odds
def best_odds(row, sportsbooks, bet_type):
    best_odds = None
    best_book = None
    for book in sportsbooks:
        col = f"{book}_{bet_type}_price"
        odds = row.get(col)

        if pd.notna(odds):
            if best_odds is None or odds > best_odds:
                best_odds = odds
                best_book = book
    return best_odds, best_book

# Load the CSV file
df = pd.read_csv('player_props_new.csv')

sportsbooks = ['fliff', 'novig', 'fanatics', 'prophetX', 'betmgm', 'bet365', 'draftkings', 'caesars', 'fanduel', 'hard_rock_bet', 'bally_bet', 'pinnacle']
over_odds = [f"{book}_over_price" for book in sportsbooks]
under_odds = [f"{book}_under_price" for book in sportsbooks]
line_columns = [f"{book}_line" for book in sportsbooks]


# --- Arbitrage Section ---

# Add new columns to the DataFrame
df['Arb %'] = None
df['Player Prop'] = None
df['Game'] = None
df['Line'] = None
df['Book 1'] = None
df['Odds Over'] = None
df['Bet Over'] = None
df['Book 2'] = None
df['Odds Under'] = None
df['Bet Under'] = None

for i, row in df.iterrows():
    # Find best odds from two books
    best_over_odds, best_over_book = best_odds(row, sportsbooks, 'over')
    best_under_odds, best_under_book = best_odds(row, sportsbooks, 'under')

    # Skip if any odds are missing
    if best_over_odds is None or best_under_odds is None:
        continue

    # Calculate if arbitrage opportunity exists
    bet_over, bet_under = calculate_arbitrage(best_over_odds, best_under_odds, 100)

    if bet_over is not None and bet_under is not None:
        # Calculate arbitrage percentage
        implied_over = implied_probability(best_over_odds)
        implied_under = implied_probability(best_under_odds)
        arb_percent = round((1 - (implied_over + implied_under)) * 100, 2)
        line_col = f"{best_over_book}_line"

        # Save the opportunity
        df.at[i, 'Arb %'] = arb_percent + '%'
        df.at[i, 'Player Prop'] = row.get('player_name', '') + " - " + row.get('market', '')
        df.at[i, 'Game'] = row.get('away_team', '') + " vs " + row.get('home_team', '')
        df.at[i, 'Line'] = row.get(line_col, '')
        df.at[i, 'Book 1'] = best_over_book
        df.at[i, 'Odds Over'] = best_over_odds
        df.at[i, 'Bet Over'] = round(bet_over, 2)
        df.at[i, 'Book 2'] = best_under_book
        df.at[i, 'Odds Under'] = best_under_odds
        df.at[i, 'Bet Under'] = round(bet_under, 2)

# Filter, Sort, and save
arb_df = df[df['Arb %'].notnull()]
arb_df = arb_df.sort_values(by='Arb %', ascending=False)
arb_df.to_csv('arbitrage_opportunities.csv', index=False)
print("Saved arbitrage opportunities to 'arbitrage_opportunities.csv'")

# --- Pos EV Section ---
