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
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))

# Convert pimplied probability to American odds
def prob_to_american_odds(prob):
    if prob > 0.5:
        return -100 * (prob / (1 - prob))
    else:
        return 100 * ((1 - prob) / prob)

# Calculate EV as a percentage
def expected_value(fair_odds, best_odds):
    win_prob = implied_probability(fair_odds)
    payout = 100 * decimal_odds(best_odds)
    edge = (win_prob * (payout - 100)) - ((1 - win_prob) * (100))
    return edge / 100 if edge > 0 else None

# Calculate Kelly Criterion as a percentage
def kelly_criterion(fair_odds, best_odds):
    win_prob = implied_probability(fair_odds)
    b = decimal_odds(best_odds) - 1
    q = 1 - win_prob
    return ((win_prob * b - q) / b) / 4 # Quarter Kelly

# Calculate no-vig (fair) odds
def no_vig(over_odds, under_odds):
    over_implied_prob = implied_probability(over_odds)
    under_implied_prob = implied_probability(under_odds)
    prob_sum = over_implied_prob + under_implied_prob 
    no_vig_over_odds = over_implied_prob / prob_sum
    no_vig_under_odds = under_implied_prob / prob_sum
    return prob_to_american_odds(no_vig_over_odds), prob_to_american_odds(no_vig_under_odds)

# Find the best odds
def best_odds(row, sportsbooks, bet_type):
    best_odds = None
    best_book = None
    for book in sportsbooks:
        col = f"{book}_{bet_type}_price"
        odds = row.get(col)

        if pd.notna(odds):
            if best_odds is None or decimal_odds(odds) > decimal_odds(best_odds):
                best_odds = odds
                best_book = book
    return best_odds, best_book

# Load the CSV file
df = pd.read_csv('/Users/sebastianbohrt/Desktop/player_props(10).csv')
arb_df = df.copy()
ev_df = df.copy()

sportsbooks = ['fliff', 'novig', 'fanatics', 'prophetX', 'betmgm', 'bet365', 'draftkings', 'caesars', 'fanduel', 'hard_rock_bet', 'bally_bet', 'pinnacle']
over_odds = [f"{book}_over_price" for book in sportsbooks]
under_odds = [f"{book}_under_price" for book in sportsbooks]
line_columns = [f"{book}_line" for book in sportsbooks]


# --- Arbitrage Section ---

# Add new columns to the DataFrame
arb_df['Arb %'] = None
arb_df['Player Prop'] = None
arb_df['Game'] = None
arb_df['Line'] = None
arb_df['Book 1'] = None
arb_df['Odds Over'] = None
arb_df['Bet Over'] = None
arb_df['Book 2'] = None
arb_df['Odds Under'] = None
arb_df['Bet Under'] = None

for i, row in df.iterrows():
    # Find best odds from two books
    best_over_odds, best_over_book = best_odds(row, sportsbooks, 'over')
    best_under_odds, best_under_book = best_odds(row, sportsbooks, 'under')
    
    # Skip if any odds are missing
    if best_over_odds is None or best_under_odds is None:
        continue

    # Get line values for best over and under books
    over_line = row.get(f"{best_over_book}_line")
    under_line = row.get(f"{best_under_book}_line")
        
    # Skip if line values are missing or don't match
    if pd.isna(over_line) or pd.isna(under_line) or over_line != under_line:
        continue

    # Calculate if arbitrage opportunity exists
    bet_over, bet_under = calculate_arbitrage(best_over_odds, best_under_odds, 100)

    if bet_over is not None and bet_under is not None:
        # Calculate arbitrage percentage
        implied_over = implied_probability(best_over_odds)
        implied_under = implied_probability(best_under_odds)
        arb_percent = round((1 - (implied_over + implied_under)) * 100, 2)

        # Save the opportunity
        arb_df.at[i, 'Arb %'] = arb_percent
        arb_df.at[i, 'Player Prop'] = row.get('player_name', '') + " - " + row.get('market', '')
        arb_df.at[i, 'Game'] = row.get('away_team', '') + " vs " + row.get('home_team', '')
        arb_df.at[i, 'Line'] = over_line
        arb_df.at[i, 'Book 1'] = best_over_book
        arb_df.at[i, 'Odds Over'] = best_over_odds
        arb_df.at[i, 'Bet Over'] = round(bet_over, 2)
        arb_df.at[i, 'Book 2'] = best_under_book
        arb_df.at[i, 'Odds Under'] = best_under_odds
        arb_df.at[i, 'Bet Under'] = round(bet_under, 2)

# Filter, Sort, and save
arb_df = arb_df[arb_df['Arb %'].notnull()]
arb_df = arb_df.sort_values(by='Arb %', ascending=False)
arb_df[['Arb %', 'Game', 'Player Prop', 'Line', 'Book 1', 'Odds Over', 'Bet Over', 'Book 2', 'Odds Under', 'Bet Under']].to_csv('Arbitrage_Opportunities.csv', index=False)
print("Saved arbitrage opportunities to 'Arbitrage_Opportunities.csv'")

# --- Pos EV Section ---
# Add new columns to the DataFrame
ev_df['EV%'] = None
ev_df['Player Prop'] = None
ev_df['Game'] = None
ev_df['Line'] = None
ev_df['Book'] = None
ev_df['Odds'] = None
ev_df['Novig Odds'] = None
ev_df['Bet Size'] = None

for i, row in df.iterrows():
    # Find best odds from two books
    best_over_odds, best_over_book = best_odds(row, sportsbooks, 'over')
    best_under_odds, best_under_book = best_odds(row, sportsbooks, 'under')

    # Find no vig odds from Pinnacle Sportsbook
    if pd.isna(row['pinnacle_over_price']) or pd.isna(row['pinnacle_under_price']):
        continue

    fair_over_odds, fair_under_odds = no_vig(row['pinnacle_over_price'], row['pinnacle_under_price'])

    if decimal_odds(best_over_odds) > decimal_odds(fair_over_odds) or decimal_odds(best_under_odds) > decimal_odds(fair_under_odds):
        # Calculate Expected Value
        ev_over = expected_value(fair_over_odds, best_over_odds)
        ev_under = expected_value(fair_under_odds, best_under_odds)

        # Calculate Expected Value
        bet_over = kelly_criterion(fair_over_odds, best_over_odds)
        bet_under = kelly_criterion(fair_under_odds, best_under_odds)

       
        # Get line values for best over and under books
        over_line = row.get(f"{best_over_book}_line")
        under_line = row.get(f"{best_under_book}_line")
        
        # Skip if line values are missing or don't match
        if pd.isna(over_line) or pd.isna(under_line) or over_line != under_line:
            continue

        # Choose between over or under line
        if ev_over and ev_under:
            # Save the opportunity
            ev_df.at[i, 'Player Prop'] = row.get('player_name', '') + " - " + row.get('market', '')
            ev_df.at[i, 'Game'] = row.get('away_team', '') + " vs " + row.get('home_team', '')
            ev_df.at[i, 'Line'] = over_line

            if ev_over > ev_under:
                ev_df.at[i, 'EV%'] = round(ev_over, 2) if ev_over else None
                ev_df.at[i, 'Book'] = best_over_book
                ev_df.at[i, 'Odds'] = best_over_odds
                ev_df.at[i, 'Novig Odds'] = round(fair_over_odds,2)
                ev_df.at[i, 'Bet Size'] = round(bet_over, 2) if bet_over else None
            else:
                ev_df.at[i, 'EV%'] = round(ev_under, 2) if ev_under else None
                ev_df.at[i, 'Book'] = best_under_book
                ev_df.at[i, 'Odds'] = best_under_odds
                ev_df.at[i, 'Novig Odds'] = round(fair_under_odds,2)
                ev_df.at[i, 'Bet Size'] = round(bet_under, 2) if bet_under else None
        elif ev_over and not ev_under:
            # Save the opportunity
            ev_df.at[i, 'Player Prop'] = row.get('player_name', '') + " - " + row.get('market', '')
            ev_df.at[i, 'Game'] = row.get('away_team', '') + " vs " + row.get('home_team', '')
            ev_df.at[i, 'Line'] = over_line
            ev_df.at[i, 'EV%'] = round(ev_over, 2)
            ev_df.at[i, 'Book'] = best_over_book
            ev_df.at[i, 'Odds'] = best_over_odds
            ev_df.at[i, 'Novig Odds'] = round(fair_over_odds,2)
            ev_df.at[i, 'Bet Size'] = round(bet_over, 2)
        elif not ev_over and ev_under:
            # Save the opportunity
            ev_df.at[i, 'Player Prop'] = row.get('player_name', '') + " - " + row.get('market', '')
            ev_df.at[i, 'Game'] = row.get('away_team', '') + " vs " + row.get('home_team', '')
            ev_df.at[i, 'Line'] = under_line
            ev_df.at[i, 'EV%'] = round(ev_under, 2)
            ev_df.at[i, 'Book'] = best_under_book
            ev_df.at[i, 'Odds'] = best_under_odds
            ev_df.at[i, 'Novig Odds'] = round(fair_under_odds,2)
            ev_df.at[i, 'Bet Size'] = round(bet_under, 2)

# Filter, Sort, and save
ev_df = ev_df[ev_df['EV%'].notnull()]
ev_df = ev_df.sort_values(by='EV%', ascending=False)
ev_df[['EV%', 'Game', 'Player Prop', 'Line', 'Book', 'Odds', 'Novig Odds', 'Bet Size']].to_csv('Expected_Value_Bets.csv', index=False)
print("Saved +EV opportunities to 'Expected_Value_Bets.csv'")

