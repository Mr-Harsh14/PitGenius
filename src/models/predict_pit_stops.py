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

def prepare_features(race_data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for prediction from race data.
    
    Args:
        race_data: DataFrame containing race data
        
    Returns:
        DataFrame containing prepared features
    """
    logger.info("Preparing features for prediction")
    
    # Set up data directories
    data_dir = Path(project_root) / 'data'
    processed_data_dir = data_dir / 'processed'
    interim_data_dir = data_dir / 'interim'
    
    # Create directories if they don't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    interim_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature engineer and data cleaner
    engineer = F1FeatureEngineer(
        processed_data_dir=str(processed_data_dir),
        interim_data_dir=str(interim_data_dir)
    )
    
    cleaner = F1DataCleaner(
        processed_data_dir=str(processed_data_dir),
        interim_data_dir=str(interim_data_dir)
    )
    
    # Clean data first
    cleaned_data = cleaner.clean_race_data(race_data)
    
    # Engineer features - apply_feature_engineering returns (transformed_df, pca_df)
    transformed_df, _ = engineer.apply_feature_engineering(cleaned_data)
    
    if transformed_df.empty:
        logger.error("Feature engineering produced empty DataFrame")
        return pd.DataFrame()
    
    # Select only the features used by the model
    model_features = [
        'TyreLife', 'DegradationPercentage', 'RollingDegradation', 'TempImpact',
        'GapToLeaderSeconds', 'RollingPaceSeconds', 'PaceConsistencySeconds',
        'UndercutPotential', 'PitWindowOptimality', 'TempSensitivity',
        'TrackEvolution', 'Position', 'PositionChange'
    ]
    
    # Ensure all required features exist
    features = pd.DataFrame(index=transformed_df.index)
    for feature in model_features:
        if feature in transformed_df.columns:
            features[feature] = transformed_df[feature]
        else:
            logger.warning(f"Feature {feature} not found in engineered features, using default value 0")
            features[feature] = 0  # Default value for missing features
    
    return features

def predict_race_pit_stops(model: 'RandomForestModel', features: pd.DataFrame) -> pd.DataFrame:
    """
    Make pit stop predictions for a race.
    
    Args:
        model: Trained Random Forest model
        features: DataFrame containing race features
        
    Returns:
        DataFrame containing predictions and probabilities
    """
    logger.info("Making pit stop predictions")
    
    # Get probabilities
    probas = model.predict_proba(features)
    
    # Create results DataFrame
    predictions = pd.DataFrame({
        'lap': features.index,
        'pit_probability': probas[:, 1],  # Probability of pit stop
        'predicted_pit': probas[:, 1] > 0.4  # Using 0.4 threshold
    })
    
    return predictions

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