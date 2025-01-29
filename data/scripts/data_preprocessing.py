"""
Data preprocessing script for F1 race data analysis.
Combines and processes lap data, driver information, and weather data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_race_data(race_path: Path):
    """
    Load all data files for a specific race.
    """
    # Load all data files
    laps_df = pd.read_csv(race_path / 'laps.csv')
    drivers_df = pd.read_csv(race_path / 'drivers.csv')
    weather_df = pd.read_csv(race_path / 'weather.csv')
    
    # Convert Driver column to string in both dataframes
    if 'Driver' in laps_df.columns:
        laps_df['Driver'] = laps_df['Driver'].astype(str)
    if 'DriverNumber' in drivers_df.columns:
        drivers_df['DriverNumber'] = drivers_df['DriverNumber'].astype(str)
    
    return laps_df, drivers_df, weather_df

def process_lap_data(laps_df):
    """
    Process lap data to extract pit stop information and lap times.
    """
    # Convert time strings to timedelta
    laps_df['LapTime'] = pd.to_timedelta(laps_df['LapTime'])
    
    # Mark pit stops
    laps_df['PitStop'] = laps_df['PitOutTime'].notna()
    
    # Calculate pit stop duration
    laps_df['PitStopDuration'] = pd.to_timedelta(laps_df['PitOutTime']) - pd.to_timedelta(laps_df['PitInTime'])
    
    # Calculate time lost in pit compared to normal lap
    pit_stop_laps = laps_df[laps_df['PitStop']]
    for idx in pit_stop_laps.index:
        driver = laps_df.loc[idx, 'Driver']
        # Get average lap time for this driver (excluding pit stop laps)
        avg_lap_time = laps_df[(laps_df['Driver'] == driver) & (~laps_df['PitStop'])]['LapTime'].mean()
        # Convert timedelta to float for calculation
        lap_time = laps_df.loc[idx, 'LapTime'].total_seconds()
        avg_time = avg_lap_time.total_seconds()
        laps_df.loc[idx, 'PitStopTimeLoss'] = pd.Timedelta(seconds=(lap_time - avg_time))
    
    # Extract compound information using ffill() instead of fillna(method='ffill')
    laps_df['TyreCompound'] = laps_df['Compound'].ffill()
    
    # Calculate stint length
    laps_df['StintLength'] = laps_df.groupby(['Driver', 'Stint'])['LapNumber'].transform('count')
    
    return laps_df

def process_weather_data(weather_df):
    """
    Process weather data to align with lap times and extract relevant features.
    """
    # Convert time strings to timedelta
    weather_df['Time'] = pd.to_timedelta(weather_df['Time'])
    
    # Calculate weather changes
    weather_df['TempChange'] = weather_df['AirTemp'].diff()
    weather_df['TrackTempChange'] = weather_df['TrackTemp'].diff()
    
    # Calculate moving averages for more stable measurements
    window = 5  # 5-minute window
    weather_df['AirTemp_MA'] = weather_df['AirTemp'].rolling(window=window).mean()
    weather_df['TrackTemp_MA'] = weather_df['TrackTemp'].rolling(window=window).mean()
    weather_df['WindSpeed_MA'] = weather_df['WindSpeed'].rolling(window=window).mean()
    
    return weather_df

def merge_data(laps_df, drivers_df, weather_df):
    """
    Merge all processed dataframes into a single analysis dataset.
    """
    # Merge driver information
    analysis_df = laps_df.merge(
        drivers_df[['DriverNumber', 'BroadcastName', 'TeamName']],
        left_on='Driver',
        right_on='DriverNumber',
        how='left'
    )
    
    # Interpolate weather data to lap times
    # Convert lap time to seconds since start for merging
    analysis_df['TimeFromStart'] = pd.to_timedelta(analysis_df['Time']).dt.total_seconds()
    weather_df['TimeFromStart'] = weather_df['Time'].dt.total_seconds()
    
    # Merge weather data based on nearest time
    analysis_df = pd.merge_asof(
        analysis_df.sort_values('TimeFromStart'),
        weather_df.sort_values('TimeFromStart'),
        on='TimeFromStart',
        direction='nearest'
    )
    
    return analysis_df

def preprocess_race_data(race_path: Path, output_path: Path):
    """
    Main function to preprocess all race data.
    """
    # Load data
    laps_df, drivers_df, weather_df = load_race_data(race_path)
    
    # Process each dataset
    laps_processed = process_lap_data(laps_df)
    weather_processed = process_weather_data(weather_df)
    
    # Merge all data
    analysis_df = merge_data(laps_processed, drivers_df, weather_processed)
    
    # Create race-specific directory in processed folder
    race_output_path = output_path / race_path.name
    os.makedirs(race_output_path, exist_ok=True)
    
    # Save processed data
    analysis_df.to_csv(race_output_path / 'race_analysis.csv', index=False)
    return analysis_df

def preprocess_all_races(base_path: Path = Path('src/data/raw/races'), 
                        output_path: Path = Path('src/data/processed')):
    """
    Preprocess all races in the dataset.
    """
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    processed_races = {}
    for race_dir in base_path.iterdir():
        if race_dir.is_dir():
            print(f"Processing {race_dir.name}...")
            try:
                processed_races[race_dir.name] = preprocess_race_data(race_dir, output_path)
                print(f"Successfully processed {race_dir.name}")
            except Exception as e:
                print(f"Failed to process {race_dir.name}: {str(e)}")
    
    return processed_races

if __name__ == "__main__":
    # Set up paths
    base_path = Path('src/data/raw/races')
    output_path = Path('src/data/processed')
    
    # Process all races
    preprocess_all_races(base_path, output_path)
