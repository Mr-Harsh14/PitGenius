import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import fastf1
import fastf1.plotting
from datetime import datetime

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.train_random_forest import load_model
from src.data.data_collector import F1DataCollector
from src.data.data_cleaner import F1DataCleaner
from src.data.feature_engineering import F1FeatureEngineer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_race_data(year: int, gp_name: str) -> pd.DataFrame:
    """
    Get race data for a specific Grand Prix.
    
    Args:
        year: Season year
        gp_name: Name of the Grand Prix
        
    Returns:
        DataFrame containing race data
    """
    logger.info(f"Getting data for {year} {gp_name}")
    
    # Enable FastF1 cache
    cache_dir = Path(project_root) / 'data' / 'raw' / 'fastf1_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    
    # Load the race
    race = fastf1.get_session(year, gp_name, 'R')
    race.load()
    
    # Get lap data with driver info
    laps_df = race.laps
    
    # Add Year and RaceNumber if not present
    laps_df['Year'] = year
    laps_df['RaceNumber'] = race.event.RoundNumber
    
    # Convert LapNumber to int if it's not already
    laps_df['LapNumber'] = laps_df['LapNumber'].astype(int)
    
    # Keep original index
    laps_df.index.name = 'Index'
    
    return laps_df

def prepare_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for prediction using the same pipeline as training.
    
    Args:
        raw_data: Raw race data
        
    Returns:
        DataFrame with engineered features
    """
    # Add required columns if missing
    required_columns = {
        'Year': None,
        'RaceNumber': None,
        'Driver': None,
        'Team': None,
        'LapNumber': None,
        'Compound': 'MEDIUM',  # Default to medium compound
        'QualyPosition': -1,   # Default to back of grid
        'DriverStandings': -1  # Default to back of standings
    }
    
    raw_data = raw_data.copy()
    for col, default_value in required_columns.items():
        if col not in raw_data.columns:
            logger.warning(f"Adding missing column: {col} with default value: {default_value}")
            raw_data[col] = default_value
    
    # Store original data we want to preserve
    original_data = raw_data[['Driver', 'LapNumber']].copy()
    
    # Initialize feature engineer
    engineer = F1FeatureEngineer(
        processed_data_dir='data/processed',
        interim_data_dir='data/interim'
    )
    
    # Process features
    transformed_df, _ = engineer.apply_feature_engineering(raw_data)
    
    # Drop target variables as we don't have them for prediction
    if 'has_pit_stop' in transformed_df.columns:
        transformed_df = transformed_df.drop(['has_pit_stop', 'good_pit_stop'], axis=1)
    
    # Handle empty DataFrame case
    if transformed_df.empty:
        logger.error("Feature engineering produced empty DataFrame")
        return pd.DataFrame()
    
    # Ensure we have features to work with
    if len(transformed_df.columns) == 0:
        logger.error("No features available after transformation")
        return pd.DataFrame()
    
    # Add back original data
    transformed_df = pd.concat([transformed_df, original_data], axis=1)
    
    logger.info(f"Prepared {len(transformed_df.columns)} features for prediction")
    return transformed_df

def predict_race_pit_stops(model, race_data: pd.DataFrame, probability_threshold: float = 0.5) -> pd.DataFrame:
    """
    Make pit stop predictions for a race.
    
    Args:
        model: Trained Random Forest model
        race_data: Race data with features
        probability_threshold: Threshold for pit stop prediction
        
    Returns:
        DataFrame with predictions including tyre compounds
    """
    try:
        # Get prediction probabilities
        if race_data.empty:
            logger.error("Cannot make predictions on empty DataFrame")
            return pd.DataFrame()
            
        # Log available columns for debugging
        logger.info(f"Available columns: {race_data.columns.tolist()}")
        
        # Get feature names from the model
        model_features = model.model.feature_names_in_
        logger.info(f"Model features: {model_features.tolist()}")
        
        # Ensure all model features are present in race_data
        missing_features = set(model_features) - set(race_data.columns)
        if missing_features:
            # For categorical features, we can add them with zeros
            # This assumes they're one-hot encoded and not present in this race
            for feature in missing_features:
                if feature.startswith(('cat__Team_', 'cat__Driver_', 'cat__Compound_')):
                    race_data[feature] = 0
                else:
                    logger.error(f"Missing non-categorical feature: {feature}")
                    return pd.DataFrame()
        
        # Select only the features used by the model
        race_data_model = race_data[model_features]
        
        # Make predictions
        pit_probs = model.model.predict_proba(race_data_model)[:, 1]
        
        # Create predictions DataFrame with original index
        predictions = pd.DataFrame(index=race_data.index)
        predictions['Driver'] = race_data['Driver']
        predictions['LapNumber'] = race_data['LapNumber']
        predictions['PitProbability'] = pit_probs
        predictions['PredictedPitStop'] = False
        
        # Get race length for strategy planning
        race_length = race_data['LapNumber'].max()
        
        # Process each driver separately
        for driver in predictions['Driver'].unique():
            driver_preds = predictions[predictions['Driver'] == driver].copy()
            
            # Find peaks above threshold
            above_threshold = driver_preds[driver_preds['PitProbability'] >= probability_threshold]
            
            if not above_threshold.empty:
                # Sort by probability to get highest peaks first
                peaks = above_threshold.sort_values('PitProbability', ascending=False)
                
                # Initialize list of selected peaks
                selected_laps = []
                
                for _, peak in peaks.iterrows():
                    # Check if this peak is far enough from all selected peaks
                    if not selected_laps or all(abs(peak['LapNumber'] - lap) >= 10 for lap in selected_laps):
                        selected_laps.append(peak['LapNumber'])
                
                # Sort selected laps by lap number
                selected_laps.sort()
                
                # Get initial compound from race data
                initial_compound = 'SOFT'  # Default to SOFT for first stint
                for compound in ['SOFT', 'MEDIUM', 'HARD']:
                    if f'cat__Compound_{compound}' in race_data.columns and \
                       race_data.loc[
                           (race_data['Driver'] == driver) & 
                           (race_data['LapNumber'] == 1),
                           f'cat__Compound_{compound}'
                       ].iloc[0] == 1:
                        initial_compound = compound
                        break
                
                # Initialize compounds array for all laps
                driver_compounds = np.full(int(race_length) + 1, initial_compound)
                
                # Plan strategy based on number of stops
                n_stops = len(selected_laps)
                
                if n_stops == 1:
                    # One-stop strategy
                    if initial_compound == 'SOFT':
                        driver_compounds[selected_laps[0]:] = 'HARD'
                    elif initial_compound == 'MEDIUM':
                        driver_compounds[selected_laps[0]:] = 'HARD'
                    else:  # HARD
                        driver_compounds[selected_laps[0]:] = 'MEDIUM'
                
                elif n_stops == 2:
                    # Two-stop strategy
                    if initial_compound == 'SOFT':
                        driver_compounds[selected_laps[0]:selected_laps[1]] = 'HARD'
                        driver_compounds[selected_laps[1]:] = 'MEDIUM'
                    elif initial_compound == 'MEDIUM':
                        driver_compounds[selected_laps[0]:selected_laps[1]] = 'HARD'
                        driver_compounds[selected_laps[1]:] = 'SOFT'
                    else:  # HARD
                        driver_compounds[selected_laps[0]:selected_laps[1]] = 'MEDIUM'
                        driver_compounds[selected_laps[1]:] = 'SOFT'
                
                else:
                    # Three or more stops - alternate between compounds
                    for i, pit_lap in enumerate(selected_laps):
                        next_pit = selected_laps[i + 1] if i + 1 < len(selected_laps) else race_length + 1
                        if driver_compounds[pit_lap - 1] == 'SOFT':
                            driver_compounds[pit_lap:next_pit] = 'HARD'
                        elif driver_compounds[pit_lap - 1] == 'HARD':
                            driver_compounds[pit_lap:next_pit] = 'MEDIUM'
                        else:  # MEDIUM
                            driver_compounds[pit_lap:next_pit] = 'SOFT'
                
                # Update predictions DataFrame
                predictions.loc[predictions['Driver'] == driver, 'CurrentCompound'] = [
                    driver_compounds[int(lap)] for lap in predictions.loc[predictions['Driver'] == driver, 'LapNumber']
                ]
                
                # Mark pit stops
                for pit_lap in selected_laps:
                    predictions.loc[
                        (predictions['Driver'] == driver) & 
                        (predictions['LapNumber'] == pit_lap),
                        'PredictedPitStop'
                    ] = True
        
        logger.info(f"Made predictions for {len(predictions)} laps")
        return predictions
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return pd.DataFrame()

def main():
    """Main function to predict pit stops for all 2024 races."""
    try:
        # Get 2024 race schedule
        schedule = fastf1.get_event_schedule(2024)
        races = schedule[schedule['EventFormat'] == 'conventional']
        
        # Load trained model
        model = load_model([2022, 2023])
        logger.info("Loaded trained model")
        
        # Process each race
        for _, race in races.iterrows():
            race_name = race['EventName']
            logger.info(f"\nProcessing {race_name}")
            
            try:
                # Get race data
                race_data = get_race_data(2024, race_name)
                logger.info(f"Got data for {len(race_data)} laps")
                
                # Prepare features
                features = prepare_features(race_data)
                logger.info("Prepared features for prediction")
                
                # Make predictions
                predictions = predict_race_pit_stops(model, features)
                logger.info("Made pit stop predictions")
                
                # Save predictions
                output_dir = Path(project_root) / 'predictions' / str(2024)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                race_name_slug = race_name.lower().replace(' ', '_')
                predictions.to_csv(output_dir / f'{race_name_slug}_predictions.csv', index=False)
                logger.info(f"Saved predictions to {output_dir / f'{race_name_slug}_predictions.csv'}")
                
            except Exception as e:
                logger.error(f"Error processing {race_name}: {str(e)}")
                continue
        
        logger.info("\nCompleted predictions for all 2024 races")
        
    except Exception as e:
        logger.error(f"Error during prediction process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 