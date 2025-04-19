import pandas as pd
import numpy as np

def extract_best_bets(
    csv_file='/Users/jamieborst/Documents/Purdue Senior Year/BQFG/arbitrage_results_with_ev_and_kelly.csv',
    arbitrage_output='/Users/jamieborst/Documents/Purdue Senior Year/BQFG/arbitrage_bets.csv',
    positive_ev_output='/Users/jamieborst/Documents/Purdue Senior Year/BQFG/positive_ev_bets.csv'
):

    def safe_float(value, default=0.0):
        try:
            if pd.isna(value) or value == '':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize lists for different bet types
    arbitrage_bets = []
    positive_ev_bets = []
    
    # Process each row to categorize bets
    for index, row in df.iterrows():
        # Check if it's an arbitrage opportunity
        over_bet = safe_float(row['Bet on Over'])
        under_bet = safe_float(row['Bet on Under'])
        
        if over_bet > 0 and under_bet > 0:
            # It's an arbitrage bet - extract only necessary fields
            arbitrage_bets.append({
                'Player_Prop': row['Player Prop'],
                'Game_Start': row['Game Start'],
                'Over_Sportsbook': row['Over Bet Sportsbook'],
                'Over_Odds': row['Best Over Odds'],
                'Over_Line': row['Best Odds Over Line'],
                'Under_Sportsbook': row['Under Bet Sportsbook'],
                'Under_Odds': row['Best Under Odds'],
                'Under_Line': row['Best Odds Under Line'],
                'Bet_On_Over': over_bet,
                'Bet_On_Under': under_bet,
                'Profit_If_Over': safe_float(row['% Profit if Over Hits']),
                'Profit_If_Under': safe_float(row['% Profit if Under Hits']),
                'Profit_If_Middle': safe_float(row['% Profit if Middle Hits'])
            })
        else:
            # Check for positive EV bets
            over_edge = safe_float(row['Over Edge (%)'])
            under_edge = safe_float(row['Under Edge (%)'])
            over_odds = safe_float(row['Best Over Odds'])
            under_odds = safe_float(row['Best Under Odds'])
            
            # Rule 2: Only consider bets with >7% edge
            # Rule 3: For over/under on same bet, take side with higher edge
            # New Rule: Don't take bets with odds over +500
            if over_edge > 7 and over_edge >= under_edge and not (over_odds > 500):
                positive_ev_bets.append({
                    'Player_Prop': row['Player Prop'],
                    'Game_Start': row['Game Start'],
                    'Bet_Side': 'Over',
                    'Sportsbook': row['Over Bet Sportsbook'],
                    'Odds': row['Best Over Odds'],
                    'Line': row['Best Odds Over Line'],
                    'Edge_Percentage': over_edge,
                    'Kelly_Bet': safe_float(row['Kelly Over Bet (% of bankroll)'])
                })
            elif under_edge > 7 and under_edge > over_edge and not (under_odds > 500):
                positive_ev_bets.append({
                    'Player_Prop': row['Player Prop'],
                    'Game_Start': row['Game Start'],
                    'Bet_Side': 'Under',
                    'Sportsbook': row['Under Bet Sportsbook'],
                    'Odds': row['Best Under Odds'],
                    'Line': row['Best Odds Under Line'],
                    'Edge_Percentage': under_edge,
                    'Kelly_Bet': safe_float(row['Kelly Under Bet (% of bankroll)'])
                })
    
    # Convert to DataFrames
    arbitrage_df = pd.DataFrame(arbitrage_bets) if arbitrage_bets else pd.DataFrame()
    positive_ev_df = pd.DataFrame(positive_ev_bets) if positive_ev_bets else pd.DataFrame()
    
    # Rule 5: If a bet is both arbitrage and positive EV, choose arbitrage
    if not arbitrage_df.empty and not positive_ev_df.empty:
        # Create a unique identifier for each bet
        arbitrage_identifiers = arbitrage_df['Player_Prop'].astype(str) + '|' + arbitrage_df['Game_Start'].astype(str)
        positive_ev_df['bet_id'] = positive_ev_df['Player_Prop'].astype(str) + '|' + positive_ev_df['Game_Start'].astype(str)
        
        # Remove bets from positive_ev_df that are also in arbitrage_df
        positive_ev_df = positive_ev_df[~positive_ev_df['bet_id'].isin(arbitrage_identifiers)]
        
        # Remove temporary column
        positive_ev_df = positive_ev_df.drop('bet_id', axis=1)
    
    # Rule 4: Max 3 positive EV bets per game
    if not positive_ev_df.empty:
        # Sort by edge in descending order
        positive_ev_df = positive_ev_df.sort_values('Edge_Percentage', ascending=False)
        
        # Group by game start time and take top 3
        positive_ev_df = positive_ev_df.groupby('Game_Start').apply(
            lambda x: x.nlargest(3, 'Edge_Percentage')).reset_index(drop=True)
    
    # Add empty result columns for tracking
    if not arbitrage_df.empty:
        arbitrage_df['Result'] = ''
        arbitrage_df['Actual_Profit'] = ''
        
    if not positive_ev_df.empty:
        positive_ev_df['Result'] = ''
        positive_ev_df['Actual_Profit'] = ''
    
    # Write to separate CSV files
    if not arbitrage_df.empty:
        arbitrage_df.to_csv(arbitrage_output, index=False)
        
    if not positive_ev_df.empty:
        positive_ev_df.to_csv(positive_ev_output, index=False)
    
    return arbitrage_df, positive_ev_df

if __name__ == "__main__":
    extract_best_bets()