import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1DataCleaner:
    """Class to clean and process Formula 1 race data."""
    
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        """
        Initialize the data cleaner.
        
        Args:
            raw_data_dir: Directory containing raw data files
            processed_data_dir: Directory to save processed data
        """
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_race_data(self, file_path: str) -> pd.DataFrame:
        """Load race data from parquet file."""
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def remove_wet_races(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove races with wet conditions."""
        wet_compounds = ['WET', 'INTERMEDIATE']
        wet_laps = df[df['Compound'].isin(wet_compounds)]
        wet_races = wet_laps[['Year', 'RaceNumber']].drop_duplicates()
        
        dry_df = df.merge(wet_races, on=['Year', 'RaceNumber'], how='left', indicator=True)
        dry_df = dry_df[dry_df['_merge'] == 'left_only'].drop('_merge', axis=1)
        
        removed_races = len(wet_races)
        logger.info(f"Removed {removed_races} wet races")
        return dry_df
    
    def clean_race_completion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data based on race completion criteria."""
        # Group by race and driver to get completion stats
        completion_stats = df.groupby(['Year', 'RaceNumber', 'Driver']).agg({
            'LapNumber': ['max', 'count'],
            'Position': 'last'
        }).reset_index()
        
        completion_stats.columns = ['Year', 'RaceNumber', 'Driver', 'LastLap', 'LapCount', 'FinalPosition']
        
        # Calculate laps behind leader for each race
        race_max_laps = completion_stats.groupby(['Year', 'RaceNumber'])['LastLap'].max().reset_index()
        completion_stats = completion_stats.merge(race_max_laps, on=['Year', 'RaceNumber'], suffixes=('', '_max'))
        completion_stats['LapsBehind'] = completion_stats['LastLap_max'] - completion_stats['LastLap']
        
        # Filter out drivers who:
        # 1. Were lapped more than 3 times
        # 2. Did not finish (significantly fewer laps than the winner)
        valid_drivers = completion_stats[
            (completion_stats['LapsBehind'] <= 3) &
            (completion_stats['LapCount'] >= completion_stats['LastLap_max'] * 0.9)
        ][['Year', 'RaceNumber', 'Driver']]
        
        # Apply filter to original dataframe
        cleaned_df = df.merge(valid_drivers, on=['Year', 'RaceNumber', 'Driver'])
        
        removed_drivers = len(df['Driver'].unique()) - len(cleaned_df['Driver'].unique())
        logger.info(f"Removed {removed_drivers} drivers who did not meet completion criteria")
        return cleaned_df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Copy dataframe to avoid modifying original
        df = df.copy()
        
        # Fill missing qualifying positions with starting grid position
        if 'QualyPosition' in df.columns:
            df['QualyPosition'] = pd.to_numeric(df['QualyPosition'], errors='coerce')
            df['QualyPosition'] = df.groupby(['Year', 'RaceNumber', 'Driver'])['QualyPosition'].transform('ffill')
        
        # Fill missing driver standings with previous value
        if 'DriverStandings' in df.columns:
            df['DriverStandings'] = pd.to_numeric(df['DriverStandings'], errors='coerce')
            df['DriverStandings'] = df.groupby(['Year', 'RaceNumber', 'Driver'])['DriverStandings'].transform('ffill')
        
        # Remove rows with missing critical values
        critical_columns = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'Position']
        initial_rows = len(df)
        df = df.dropna(subset=critical_columns)
        removed_rows = initial_rows - len(df)
        
        logger.info(f"Removed {removed_rows} rows with missing critical values")
        return df
    
    def clean_race_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all cleaning steps to the race data."""
        logger.info(f"Starting data cleaning with {len(df)} rows")
        
        # Remove wet races
        df = self.remove_wet_races(df)
        logger.info(f"After removing wet races: {len(df)} rows")
        
        # Clean based on race completion
        df = self.clean_race_completion(df)
        logger.info(f"After cleaning race completion: {len(df)} rows")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        logger.info(f"After handling missing values: {len(df)} rows")
        
        return df
    
    def process_all_files(self) -> None:
        """Process all raw data files and save cleaned versions."""
        try:
            # Get all parquet files in raw data directory
            raw_files = list(self.raw_data_dir.glob("*.parquet"))
            
            if not raw_files:
                logger.warning(f"No parquet files found in {self.raw_data_dir}")
                return
            
            # Process each file
            for file_path in raw_files:
                logger.info(f"Processing {file_path}")
                
                # Load data
                df = self.load_race_data(file_path)
                if df.empty:
                    continue
                
                # Clean data
                cleaned_df = self.clean_race_data(df)
                
                # Save processed data
                output_path = self.processed_data_dir / f"cleaned_{file_path.name}"
                cleaned_df.to_parquet(output_path)
                logger.info(f"Saved cleaned data to {output_path}")
                
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")

def main():
    """Main function to run the data cleaning process."""
    # Set up directories
    raw_data_dir = "data/raw/fastf1_cache"
    processed_data_dir = "data/processed"
    
    # Initialize cleaner
    cleaner = F1DataCleaner(raw_data_dir, processed_data_dir)
    
    # Run cleaning process
    cleaner.process_all_files()

if __name__ == "__main__":
    main()
