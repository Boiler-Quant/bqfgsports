import json
import boto3
import pandas as pd
import numpy as np
import logging
from io import StringIO
import os

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
            if best_odds is None or odds > best_odds:
                best_odds = odds
                best_book = book
    return best_odds, best_book

def calculate_arbitrage_opportunities(df):
    """Calculate arbitrage opportunities in player props data."""
    # Create a copy for arbitrage calculation
    arb_df = df.copy()
    
    # List of sportsbooks - using the ones from the original code
    sportsbooks = ['betmgm', 'betonlineag', 'betrivers', 'bovada', 'draftkings', 'fanduel']
    
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
    
    # Filter out rows with no arbitrage opportunities
    arb_df = arb_df[arb_df['Arb %'].notnull()]
    
    # Sort by best arbitrage opportunities (descending)
    arb_df = arb_df.sort_values(by='Arb %', ascending=False)
    
    # Return only the columns we need
    return arb_df[['Arb %', 'Player Prop', 'Game', 'Line', 'Book 1', 'Odds Over', 'Bet Over', 'Book 2', 'Odds Under', 'Bet Under']]
    
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
                        
                        # Save arbitrage results to S3
                        output_key = 'final/arbitrage_results_with_ev_and_kelly.csv'
                        csv_buffer = StringIO()
                        arb_df.to_csv(csv_buffer, index=False)
                        
                        s3_client.put_object(
                            Bucket=s3_bucket,
                            Key=output_key,
                            Body=csv_buffer.getvalue()
                        )
                        
                        logger.info(f"Successfully calculated arbitrage opportunities and saved to {output_key}")
                        
                        return {
                            'statusCode': 200,
                            'body': json.dumps('Successfully processed arbitrage opportunities')
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
                
                # Save arbitrage results to S3
                output_key = 'final/arbitrage_results_with_ev_and_kelly.csv'
                csv_buffer = StringIO()
                arb_df.to_csv(csv_buffer, index=False)
                
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=output_key,
                    Body=csv_buffer.getvalue()
                )
                
                logger.info(f"Successfully calculated arbitrage opportunities and saved to {output_key}")
                
                return {
                    'statusCode': 200,
                    'body': json.dumps('Successfully processed arbitrage opportunities')
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
