import json
import boto3
import pandas as pd
import numpy as np
import logging
from io import StringIO
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# --- Arbitrage Functions ---

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
    if pd.isna(decimal_odds):
        return None
    
    # Handle edge cases
    if decimal_odds <= 1.0:
        # Invalid odds, return None
        return None
    
    if decimal_odds >= 2.0:
        # For decimal odds >= 2.0, American odds are positive
        return round((decimal_odds - 1) * 100)
    else:
        # For decimal odds < 2.0, American odds are negative
        return round(-100 / (decimal_odds - 1))

# Convert implied probability to American odds
def prob_to_american_odds(prob):
    if prob > 0.5:
        return -100 * (prob / (1 - prob))
    else:
        return 100 * ((1 - prob) / prob)

# Calculate no-vig (fair) odds
def no_vig(over_odds, under_odds):
    over_implied_prob = implied_probability(over_odds)
    under_implied_prob = implied_probability(under_odds)
    sum_prob = over_implied_prob + under_implied_prob 
    no_vig_over_odds = over_implied_prob / sum_prob
    no_vig_under_odds = under_implied_prob / sum_prob
    return prob_to_american_odds(no_vig_over_odds), prob_to_american_odds(no_vig_under_odds)

# Calculate EV as a percentage - updated to match the approach in paste-2.txt
def expected_value(fair_odds, best_odds):
    win_prob = implied_probability(fair_odds)
    payout = 100 * decimal_odds(best_odds)
    edge = (win_prob * (payout - 100)) - ((1 - win_prob) * (100))
    return edge / 100 if edge > 0 else None

# Calculate Kelly Criterion as a percentage - updated to use quarter Kelly
def kelly_criterion(fair_odds, best_odds):
    win_prob = implied_probability(fair_odds)
    b = decimal_odds(best_odds) - 1
    q = 1 - win_prob
    return ((win_prob * b - q) / b) / 4  # Quarter Kelly instead of full Kelly

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

def calculate_arbitrage_opportunities(df):
    """Calculate arbitrage opportunities in player props data."""
    # Create a copy for arbitrage calculation
    arb_df = df.copy()
    
    # List of sportsbooks - using the ones from the original code
    sportsbooks = ['fliff', 'fanatics', 'prophetX', 'betmgm', 'bet365', 'draftkings', 'caesars', 'fanduel', 'hard_rock_bet', 'bally_bet']
    
    # Initialize new columns for arbitrage DataFrame
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
    arb_df['timestamp'] = datetime.now().strftime('%m:%d:%y:%H:%M:%S.%f')
    
    # Process each row for arbitrage opportunities
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
            
            # Save the arbitrage opportunity
            arb_df.at[i, 'Arb %'] = arb_percent
            arb_df.at[i, 'Player Prop'] = row.get('player_name', '') + " - " + row.get('market', '')
            arb_df.at[i, 'Game'] = row.get('away_team', '') + " vs " + row.get('home_team', '')
            arb_df.at[i, 'Line'] = over_line  # Using the matching line value
            arb_df.at[i, 'Book 1'] = best_over_book
            arb_df.at[i, 'Odds Over'] = best_over_odds
            arb_df.at[i, 'Bet Over'] = round(bet_over, 2)
            arb_df.at[i, 'Book 2'] = best_under_book
            arb_df.at[i, 'Odds Under'] = best_under_odds
            arb_df.at[i, 'Bet Under'] = round(bet_under, 2)
            arb_df.at[i, 'bet_id'] = f"{row.get('game_id')}_{row.get('market')}_{row.get('player_name')}_{over_line}"
    
    # Filter out rows with no arbitrage opportunities
    arb_df = arb_df[arb_df['Arb %'].notnull()]
    
    # Sort by best arbitrage opportunities (descending)
    arb_df = arb_df.sort_values(by='Arb %', ascending=False)
    
    # Return only the columns we need
    return arb_df[['Arb %', 'Player Prop', 'Game', 'Line', 'Book 1', 'Odds Over', 'Bet Over', 'Book 2', 'Odds Under', 'Bet Under', 'timestamp', 'bet_id']]

def calculate_ev_opportunities(df):
    """Calculate positive EV opportunities in player props data - updated to use new approach."""
    # Create a copy for EV calculation
    ev_df = df.copy()
    
    # List of sportsbooks - expanded to match the updated list
    sportsbooks = ['fliff', 'fanatics', 'prophetX', 'betmgm', 'bet365', 'draftkings', 'caesars', 'fanduel', 'hard_rock_bet', 'bally_bet', 'pinnacle']

    
    # Initialize new columns for EV DataFrame
    ev_df['EV %'] = None
    ev_df['Player Prop'] = None
    ev_df['Game'] = None
    ev_df['Line'] = None
    ev_df['Book'] = None
    ev_df['Odds'] = None
    ev_df['Novig Odds'] = None
    ev_df['Bet Size'] = None
    ev_df['Bet Type'] = None
    ev_df['timestamp'] = datetime.now().strftime('%m:%d:%y:%H:%M:%S.%f')
    
    # Process each row for EV opportunities
    for i, row in df.iterrows():
        # Skip if Pinnacle odds are missing
        if pd.isna(row.get('pinnacle_over_price')) or pd.isna(row.get('pinnacle_under_price')):
            continue
            
        # Calculate fair odds using Pinnacle as the reference book
        fair_over_odds, fair_under_odds = no_vig(
            row.get('pinnacle_over_price'),
            row.get('pinnacle_under_price')
        )
        
        # Find best over odds from all books
        best_over_odds, best_over_book = best_odds(row, sportsbooks, 'over')
        
        # Find best under odds from all books
        best_under_odds, best_under_book = best_odds(row, sportsbooks, 'under')
        
        # Skip if any odds are missing
        if best_over_odds is None or best_under_odds is None:
            continue
        
        # Get line values for best odds books
        over_line = row.get(f"{best_over_book}_line")
        under_line = row.get(f"{best_under_book}_line")
        
        # Skip if line values are missing or don't match
        if pd.isna(over_line) or pd.isna(under_line) or over_line != under_line:
            continue
            
        # Check if there is value in either over or under bet
        if decimal_odds(best_over_odds) > decimal_odds(fair_over_odds) or decimal_odds(best_under_odds) > decimal_odds(fair_under_odds):
            # Calculate EV for over and under bets
            ev_over = expected_value(fair_over_odds, best_over_odds)
            ev_under = expected_value(fair_under_odds, best_under_odds)
            
            # Calculate Kelly criterion for bet sizing
            kelly_over = kelly_criterion(fair_over_odds, best_over_odds)
            kelly_under = kelly_criterion(fair_under_odds, best_under_odds)
            
            # Process both over and under opportunities
            if ev_over is not None and ev_under is not None:
                # Choose the better EV between over and under
                if ev_over > ev_under:
                    ev_df.at[i, 'EV %'] = round(ev_over * 100, 2)
                    ev_df.at[i, 'Player Prop'] = row.get('player_name', '') + " - " + row.get('market', '')
                    ev_df.at[i, 'Game'] = row.get('away_team', '') + " vs " + row.get('home_team', '')
                    ev_df.at[i, 'Line'] = over_line
                    ev_df.at[i, 'Book'] = best_over_book
                    ev_df.at[i, 'Odds'] = best_over_odds
                    ev_df.at[i, 'Novig Odds'] = round(fair_over_odds, 2)
                    ev_df.at[i, 'Bet Size'] = round(kelly_over * 100, 2) if kelly_over else None
                    ev_df.at[i, 'Bet Type'] = 'Over'
                    ev_df.at[i, 'bet_id'] = f"{row.get('game_id')}_{row.get('market')}_{row.get('player_name')}_{over_line}_over"
                else:
                    ev_df.at[i, 'EV %'] = round(ev_under * 100, 2)
                    ev_df.at[i, 'Player Prop'] = row.get('player_name', '') + " - " + row.get('market', '')
                    ev_df.at[i, 'Game'] = row.get('away_team', '') + " vs " + row.get('home_team', '')
                    ev_df.at[i, 'Line'] = under_line
                    ev_df.at[i, 'Book'] = best_under_book
                    ev_df.at[i, 'Odds'] = best_under_odds
                    ev_df.at[i, 'Novig Odds'] = round(fair_under_odds, 2)
                    ev_df.at[i, 'Bet Size'] = round(kelly_under * 100, 2) if kelly_under else None
                    ev_df.at[i, 'Bet Type'] = 'Under'
                    ev_df.at[i, 'bet_id'] = f"{row.get('game_id')}_{row.get('market')}_{row.get('player_name')}_{under_line}_under"
            # If only over EV exists
            elif ev_over is not None:
                ev_df.at[i, 'EV %'] = round(ev_over * 100, 2)
                ev_df.at[i, 'Player Prop'] = row.get('player_name', '') + " - " + row.get('market', '')
                ev_df.at[i, 'Game'] = row.get('away_team', '') + " vs " + row.get('home_team', '')
                ev_df.at[i, 'Line'] = over_line
                ev_df.at[i, 'Book'] = best_over_book
                ev_df.at[i, 'Odds'] = best_over_odds
                ev_df.at[i, 'Novig Odds'] = round(fair_over_odds, 2)
                ev_df.at[i, 'Bet Size'] = round(kelly_over * 100, 2) if kelly_over else None
                ev_df.at[i, 'Bet Type'] = 'Over'
                ev_df.at[i, 'bet_id'] = f"{row.get('game_id')}_{row.get('market')}_{row.get('player_name')}_{over_line}_over"
            # If only under EV exists
            elif ev_under is not None:
                ev_df.at[i, 'EV %'] = round(ev_under * 100, 2)
                ev_df.at[i, 'Player Prop'] = row.get('player_name', '') + " - " + row.get('market', '')
                ev_df.at[i, 'Game'] = row.get('away_team', '') + " vs " + row.get('home_team', '')
                ev_df.at[i, 'Line'] = under_line
                ev_df.at[i, 'Book'] = best_under_book
                ev_df.at[i, 'Odds'] = best_under_odds
                ev_df.at[i, 'Novig Odds'] = round(fair_under_odds, 2)
                ev_df.at[i, 'Bet Size'] = round(kelly_under * 100, 2) if kelly_under else None
                ev_df.at[i, 'Bet Type'] = 'Under'
                ev_df.at[i, 'bet_id'] = f"{row.get('game_id')}_{row.get('market')}_{row.get('player_name')}_{under_line}_under"
    
    # Filter out rows with no EV opportunities
    ev_df = ev_df[ev_df['EV %'].notnull()]
    
    # Sort by best EV opportunities (descending)
    ev_df = ev_df.sort_values(by='EV %', ascending=False)
    
    # Return only the columns we need
    return ev_df[['EV %', 'Player Prop', 'Game', 'Line', 'Book', 'Odds', 'Novig Odds', 'Bet Size', 'Bet Type', 'timestamp', 'bet_id']]

def update_bet_history(arb_df, ev_df, s3_client, bucket_name):
    """Update the bet history by combining active bet information."""
    # Combine arbitrage and EV opportunities
    combined_df = pd.DataFrame()
    
    if not arb_df.empty:
        arb_df['Type'] = 'Arbitrage'
        combined_df = pd.concat([combined_df, arb_df], ignore_index=True)
        
    if not ev_df.empty:
        ev_df['Type'] = 'Expected Value'
        combined_df = pd.concat([combined_df, ev_df], ignore_index=True)
    
    if combined_df.empty:
        logger.info("No bets to track history for.")
        return
        
    # Try to read existing active bets from S3
    try:
        response = s3_client.get_object(
            Bucket=bucket_name,
            Key='active/active_bets.csv'
        )
        active_df = pd.read_csv(response['Body'])
        active_df['timestamp'] = pd.to_datetime(active_df['timestamp'], errors='coerce')
    except Exception as e:
        logger.info(f"No active bets file found or error reading file: {e}")
        active_df = pd.DataFrame(columns=combined_df.columns.tolist() + ['duration'])
    
    # Add duration column if not present
    if 'duration' not in active_df.columns:
        active_df['duration'] = 0
        
    # Get current timestamp
    current_time = datetime.now()
    
    # Create dictionaries for faster lookups
    active_dict = {row['bet_id']: row for _, row in active_df.iterrows()} if 'bet_id' in active_df else {}
    
    # Prepare updated active and expired bets lists
    updated_active = []
    expired_bets = []
    
    # Process all new bets
    for _, new_row in combined_df.iterrows():
        if 'bet_id' not in new_row or pd.isna(new_row['bet_id']):
            continue
            
        bet_id = new_row['bet_id']
        
        if bet_id in active_dict:
            # Calculate duration for existing bet
            active_start = pd.to_datetime(active_dict[bet_id]['timestamp'])
            new_duration = (current_time - active_start).total_seconds()
            
            # Update duration
            new_row = new_row.copy()
            new_row['duration'] = new_duration
            updated_active.append(new_row)
            
            # Remove from active dict to track what's no longer active
            del active_dict[bet_id]
        else:
            # Add new bet with 0 duration
            new_row = new_row.copy()
            new_row['duration'] = 0
            updated_active.append(new_row)
    
    # Process expired bets
    for bet_id, expired_row in active_dict.items():
        # Calculate final duration
        start_time = pd.to_datetime(expired_row['timestamp'])
        expired_row['duration'] = (current_time - start_time).total_seconds()
        expired_bets.append(expired_row)
    
    # Convert to DataFrames
    updated_active_df = pd.DataFrame(updated_active)
    
    # Save updated active bets to S3
    if not updated_active_df.empty:
        csv_buffer = StringIO()
        updated_active_df.to_csv(csv_buffer, index=False)
        s3_client.put_object(
            Bucket=bucket_name,
            Key='active/active_bets.csv',
            Body=csv_buffer.getvalue()
        )
        logger.info(f"Updated active bets file with {len(updated_active_df)} bets")
    
    # Handle expired bets - add to history
    if expired_bets:
        expired_df = pd.DataFrame(expired_bets)
        try:
            response = s3_client.get_object(
                Bucket=bucket_name,
                Key='history/bet_history.csv'
            )
            history_df = pd.read_csv(response['Body'])
            history_df = pd.concat([history_df, expired_df], ignore_index=True)
        except Exception as e:
            logger.info(f"No history file found or error reading file: {e}")
            history_df = expired_df
        
        # Save updated history to S3
        csv_buffer = StringIO()
        history_df.to_csv(csv_buffer, index=False)
        s3_client.put_object(
            Bucket=bucket_name,
            Key='history/bet_history.csv',
            Body=csv_buffer.getvalue()
        )
        logger.info(f"Added {len(expired_bets)} expired bets to history")
    
    return

def lambda_handler(event, context):
    try:
        # Get S3 bucket name from environment variable
        s3_bucket = os.environ['S3_BUCKET_NAME']
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Check if event is from S3
        if 'Records' in event:
            # Get file info from S3 event
            for record in event['Records']:
                if record['eventSource'] == 'aws:s3':
                    file_key = record['s3']['object']['key']
                    logger.info(f"Processing file: {file_key}")
                    
                    if 'player_props.csv' in file_key:
                        # Download the file from S3
                        response = s3_client.get_object(
                            Bucket=s3_bucket,
                            Key=file_key
                        )
                        
                        # Read the CSV into a DataFrame
                        df = pd.read_csv(response['Body'])
                        
                        # Calculate arbitrage opportunities
                        arb_df = calculate_arbitrage_opportunities(df)
                        
                        # Calculate EV opportunities using new approach
                        ev_df = calculate_ev_opportunities(df)
                        
                        # Update bet history
                        update_bet_history(arb_df, ev_df, s3_client, s3_bucket)
                        
                        # Save arbitrage results to S3
                        arb_output_key = 'final/arbitrage_results_with_ev_and_kelly.csv'
                        arb_csv_buffer = StringIO()
                        arb_df.to_csv(arb_csv_buffer, index=False)
                        
                        s3_client.put_object(
                            Bucket=s3_bucket,
                            Key=arb_output_key,
                            Body=arb_csv_buffer.getvalue()
                        )
                        
                        # Save EV results to S3
                        ev_output_key = 'final/ev_results.csv'
                        ev_csv_buffer = StringIO()
                        ev_df.to_csv(ev_csv_buffer, index=False)
                        
                        s3_client.put_object(
                            Bucket=s3_bucket,
                            Key=ev_output_key,
                            Body=ev_csv_buffer.getvalue()
                        )
                        
                        logger.info(f"Successfully calculated arbitrage and EV opportunities and saved to S3")
                        
                        return {
                            'statusCode': 200,
                            'body': json.dumps('Successfully processed arbitrage and EV opportunities')
                        }
                    else:
                        logger.info(f"File {file_key} is not player_props.csv, skipping")
        else:
            # If not triggered by S3, try to get the file from raw folder
            try:
                response = s3_client.get_object(
                    Bucket=s3_bucket,
                    Key='raw/player_props.csv'
                )
                
                # Read the CSV into a DataFrame
                df = pd.read_csv(response['Body'])
                
                # Calculate arbitrage opportunities
                arb_df = calculate_arbitrage_opportunities(df)
                
                # Calculate EV opportunities using new approach
                ev_df = calculate_ev_opportunities(df)
                
                # Update bet history
                update_bet_history(arb_df, ev_df, s3_client, s3_bucket)
                
                # Save arbitrage results to S3
                arb_output_key = 'final/arbitrage_results_with_ev_and_kelly.csv'
                arb_csv_buffer = StringIO()
                arb_df.to_csv(arb_csv_buffer, index=False)
                
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=arb_output_key,
                    Body=arb_csv_buffer.getvalue()
                )
                
                # Save EV results to S3
                ev_output_key = 'final/ev_results.csv'
                ev_csv_buffer = StringIO()
                ev_df.to_csv(ev_csv_buffer, index=False)
                
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=ev_output_key,
                    Body=ev_csv_buffer.getvalue()
                )
                
                logger.info(f"Successfully calculated arbitrage and EV opportunities and saved to S3")
                
                return {
                    'statusCode': 200,
                    'body': json.dumps('Successfully processed arbitrage and EV opportunities')
                }
            except Exception as e:
                logger.error(f"Error retrieving or processing file: {str(e)}")
                raise e
        
        return {
            'statusCode': 200,
            'body': json.dumps('No appropriate file to process')
        }
        
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }
