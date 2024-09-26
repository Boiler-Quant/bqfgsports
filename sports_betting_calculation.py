import pandas as pd

# Load the CSV file
file_path = '/Users/jamieborst/Documents/Purdue Senior Year/BQFG/player_props.csv'
df = pd.read_csv(file_path)

# Function to calculate arbitrage opportunity
def calculate_arbitrage(over_odds, under_odds, over_line, under_line):
    over_implied_prob = 100 / (over_odds + 100) if over_odds > 0 else -over_odds / (-over_odds + 100)
    under_implied_prob = 100 / (under_odds + 100) if under_odds > 0 else -under_odds / (-under_odds + 100)
    
    if over_implied_prob + under_implied_prob < 1:
        total_investment = 1 / (over_implied_prob + under_implied_prob)
        bet_on_over = (over_implied_prob * total_investment)
        bet_on_under = (under_implied_prob * total_investment)
        
        # Calculate the payouts
        payout_over = bet_on_over * (1 + (over_odds / 100 if over_odds > 0 else 100 / -over_odds))
        payout_under = bet_on_under * (1 + (under_odds / 100 if under_odds > 0 else 100 / -under_odds))

        # Calculate the profits
        profit_over = payout_over - (bet_on_over + bet_on_under)  # total bet amount
        profit_under = payout_under - (bet_on_over + bet_on_under)  # total bet amount
        
        # Calculate middle hit profit if applicable
        middle_profit = 0
        if over_line != under_line:  # Check if the lines are different
            middle_profit = profit_over + profit_under

        return bet_on_over, bet_on_under, profit_over, profit_under, middle_profit
    else:
        return None, None, None, None, None

# Function to calculate EV (Expected Value)
def calculate_ev(best_odds, consensus_odds):
    if best_odds > 0:  # Positive odds
        implied_prob_best = 100 / (best_odds + 100)
    else:  # Negative odds
        implied_prob_best = -best_odds / (-best_odds + 100)
    
    if consensus_odds > 0:  # Positive odds
        implied_prob_consensus = 100 / (consensus_odds + 100)
    else:  # Negative odds
        implied_prob_consensus = -consensus_odds / (-consensus_odds + 100)
    
    # Calculate the edge
    edge = implied_prob_consensus - implied_prob_best
    
    # Calculate expected value
    ev = edge * (1 / implied_prob_best) if edge > 0 else 0
    
    return edge, ev

# Function to convert American odds to implied probability
def american_odds_to_probability(odds):
    if odds < 0:
        return -odds / (-odds + 100)
    else:
        return 100 / (odds + 100)

# Function to calculate Kelly Criterion
def kelly_criterion(prob, odds):
    decimal_odds = (odds / 100) + 1 if odds > 0 else (-100 / odds) + 1
    edge = prob - (1 / decimal_odds)
    return edge / (decimal_odds - 1) if edge > 0 else 0

# Convert the odds from string to integer
df['Over Odds'] = df['Over Best Odds Value'].str.extract(r'([-+]?\d+)').astype(int)
df['Under Odds'] = df['Under Best Odds Value'].str.extract(r'([-+]?\d+)').astype(int)

# Convert the over and under lines from string to float
df['Over Line'] = df['Over Best Odds'].str.extract(r'(\d+\.\d+|\d+)').astype(float)
df['Under Line'] = df['Under Best Odds'].str.extract(r'(\d+\.\d+|\d+)').astype(float)

# Extract consensus odds values into new columns for easy access
df['Over Consensus Odds'] = df['Over Consensus Odds Value'].str.extract(r'([-+]?\d+)').astype(int)
df['Under Consensus Odds'] = df['Under Consensus Odds Value'].str.extract(r'([-+]?\d+)').astype(int)

# List to hold the results
results = []

# Calculate arbitrage opportunities, EV, and Kelly Criterion
for index, row in df.iterrows():
    # Arbitrage calculation
    bet_on_over, bet_on_under, profit_over, profit_under, middle_profit = calculate_arbitrage(
        row['Over Odds'], 
        row['Under Odds'], 
        row['Over Line'], 
        row['Under Line']
    )

    # Calculate EV and edge for Over
    _, ev_over = calculate_ev(row['Over Odds'], row['Over Consensus Odds'])
    
    # Calculate EV and edge for Under
    _, ev_under = calculate_ev(row['Under Odds'], row['Under Consensus Odds'])

    # Calculate Kelly Criterion for Over and Under
    over_prob = american_odds_to_probability(row['Over Consensus Odds'])
    under_prob = american_odds_to_probability(row['Under Consensus Odds'])
    kelly_over = kelly_criterion(over_prob, row['Over Odds'])
    kelly_under = kelly_criterion(under_prob, row['Under Odds'])

    # Append the results to the list
    results.append({
        'Player Prop': row['Player Prop'],
        'Bet on Over': round(bet_on_over * 100, 2) if bet_on_over is not None else '',
        'Bet on Under': round(bet_on_under * 100, 2) if bet_on_under is not None else '',
        'Profit if Over Hits': round(profit_over * 100, 2) if profit_over is not None else '',
        'Profit if Under Hits': round(profit_under * 100, 2) if profit_under is not None else '',
        'Profit if Middle Hits': round(middle_profit * 100, 2) if middle_profit is not None else '',
        'Over EV (%)': round(ev_over * 100, 2) if ev_over else '',
        'Under EV (%)': round(ev_under * 100, 2) if ev_under else '',
        'Kelly Over Bet (%)': round(kelly_over * 100, 2) if kelly_over else '',
        'Kelly Under Bet (%)': round(kelly_under * 100, 2) if kelly_under else ''
    })

# Create a DataFrame from the results and save it to a CSV file
output_df = pd.DataFrame(results)
output_df.to_csv('/Users/jamieborst/Documents/Purdue Senior Year/BQFG/arbitrage_results_with_ev_and_kelly.csv', index=False)

print("Arbitrage, EV, and Kelly Criterion calculations have been saved to 'arbitrage_results_with_ev_and_kelly.csv'.")
