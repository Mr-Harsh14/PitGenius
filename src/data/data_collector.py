import os
import logging
import pandas as pd
import fastf1
import requests
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1DataCollector:
    """Class to collect Formula 1 data from FastF1 and Ergast API."""
    
    def __init__(self, cache_dir: str, seasons: List[int]):
        """
        Initialize the data collector.
        
        Args:
            cache_dir: Directory to store FastF1 cache
            seasons: List of seasons to collect data from
        """
        self.cache_dir = Path(cache_dir)
        self.seasons = seasons
        self.ergast_base_url = "http://ergast.com/api/f1"
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory ensured at {self.cache_dir}")
        
        # Enable FastF1 cache
        fastf1.Cache.enable_cache(str(self.cache_dir))
        logger.info(f"FastF1 cache enabled at {self.cache_dir}")

    def get_race_schedule(self, year: int) -> pd.DataFrame:
        """Get the race schedule for a specific year."""
        try:
            schedule = fastf1.get_event_schedule(year)
            # Filter out non-race events (testing, etc.)
            schedule = schedule[
                (schedule['EventFormat'] == 'conventional') & 
                (schedule['RoundNumber'] > 0)
            ]
            logger.info(f"Retrieved {len(schedule)} races for {year}")
            return schedule
        except Exception as e:
            logger.error(f"Error getting race schedule for {year}: {str(e)}")
            return pd.DataFrame()

    def get_race_data(self, year: int, race_round: int) -> Dict[str, Any]:
        """
        Get detailed race data for a specific race.
        
        Args:
            year: Season year
            race_round: Race round number
        
        Returns:
            Dictionary containing race data
        """
        try:
            # Load the race session
            race = fastf1.get_session(year, race_round, 'R')
            race.load()
            
            # Get lap data for all drivers
            laps_data = race.laps
            
            # Get driver info and ensure DriverNumber is present
            drivers_data = pd.DataFrame(race.drivers)
            if 'DriverNumber' not in drivers_data.columns:
                drivers_data['DriverNumber'] = drivers_data.index
            
            # Get weather data
            weather_data = race.weather_data
            
            return {
                'laps': laps_data,
                'drivers': drivers_data,
                'weather': weather_data
            }
        except Exception as e:
            logger.error(f"Error getting race data for {year} round {race_round}: {str(e)}")
            return {}

    def get_ergast_data(self, year: int, race_round: int) -> Dict[str, Any]:
        """
        Get data from Ergast API for a specific race.
        
        Args:
            year: Season year
            race_round: Race round number
        
        Returns:
            Dictionary containing Ergast API data
        """
        try:
            # Get qualifying results
            quali_url = f"{self.ergast_base_url}/{year}/{race_round}/qualifying.json"
            quali_response = requests.get(quali_url)
            quali_data = quali_response.json()
            
            # Get driver standings
            standings_url = f"{self.ergast_base_url}/{year}/{race_round}/driverStandings.json"
            standings_response = requests.get(standings_url)
            standings_data = standings_response.json()
            
            return {
                'qualifying': quali_data,
                'standings': standings_data
            }
        except Exception as e:
            logger.error(f"Error getting Ergast data for {year} round {race_round}: {str(e)}")
            return {}

    def process_race_data(self, fastf1_data: Dict[str, Any], ergast_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Process and combine data from both sources.
        
        Args:
            fastf1_data: Data from FastF1
            ergast_data: Data from Ergast API
        
        Returns:
            DataFrame containing processed race data
        """
        try:
            if not fastf1_data or 'laps' not in fastf1_data:
                return pd.DataFrame()
                
            # Extract lap data
            laps_df = fastf1_data['laps'].copy()
            
            # Add driver info
            if 'drivers' in fastf1_data and not fastf1_data['drivers'].empty:
                drivers_info = fastf1_data['drivers']
                # Ensure DriverNumber is string type in both DataFrames
                laps_df['DriverNumber'] = laps_df['DriverNumber'].astype(str)
                drivers_info['DriverNumber'] = drivers_info['DriverNumber'].astype(str)
                laps_df = laps_df.merge(
                    drivers_info,
                    on='DriverNumber',
                    how='left'
                )
            
            # Add qualifying positions from Ergast
            if ergast_data.get('qualifying'):
                quali_data = pd.json_normalize(
                    ergast_data['qualifying']['MRData']['RaceTable']['Races'][0]['QualifyingResults']
                )
                # Use Driver.code for consistent driver identification
                quali_data = quali_data.rename(columns={'Driver.code': 'Driver'})
                # Ensure Driver column exists in laps_df
                if 'Driver' in laps_df.columns:
                    laps_df = laps_df.merge(
                        quali_data[['Driver', 'position']],
                        on='Driver',
                        how='left'
                    ).rename(columns={'position': 'QualyPosition'})
            
            # Add driver standings
            if ergast_data.get('standings'):
                standings_data = pd.json_normalize(
                    ergast_data['standings']['MRData']['StandingsTable']['StandingsLists'][0]['DriverStandings']
                )
                # Use Driver.code for consistent driver identification
                standings_data = standings_data.rename(columns={'Driver.code': 'Driver'})
                # Ensure Driver column exists in laps_df
                if 'Driver' in laps_df.columns:
                    laps_df = laps_df.merge(
                        standings_data[['Driver', 'position']],
                        on='Driver',
                        how='left'
                    ).rename(columns={'position': 'DriverStandings'})
            
            return laps_df
        except Exception as e:
            logger.error(f"Error processing race data: {str(e)}")
            return pd.DataFrame()

    def collect_season_data(self, year: int) -> None:
        """
        Collect data for an entire season.
        
        Args:
            year: Season year
        """
        try:
            # Get race schedule
            schedule = self.get_race_schedule(year)
            if schedule.empty:
                return
            
            all_race_data = []
            for _, race in schedule.iterrows():
                race_round = race['RoundNumber']
                race_name = race['EventName']
                
                logger.info(f"Collecting data for {year} {race_name}")
                
                # Get data from both sources
                fastf1_data = self.get_race_data(year, race_round)
                ergast_data = self.get_ergast_data(year, race_round)
                
                # Process and combine data
                race_df = self.process_race_data(fastf1_data, ergast_data)
                if not race_df.empty:
                    race_df['Year'] = year
                    race_df['RaceNumber'] = race_round
                    race_df['RaceName'] = race_name
                    all_race_data.append(race_df)
            
            if all_race_data:
                # Combine all races and save
                season_df = pd.concat(all_race_data, ignore_index=True)
                output_path = self.cache_dir / f"season_{year}.parquet"
                season_df.to_parquet(output_path)
                logger.info(f"Saved {year} season data to {output_path}")
                
        except Exception as e:
            logger.error(f"Error collecting season data for {year}: {str(e)}")

    def collect_all_seasons(self) -> None:
        """Collect data for all specified seasons."""
        for season in self.seasons:
            logger.info(f"Starting data collection for {season} season")
            self.collect_season_data(season)
            logger.info(f"Completed data collection for {season} season")

def main():
    """Main function to run the data collection process."""
    # Load configuration
    cache_dir = os.getenv('FASTF1_CACHE_DIR', 'data/raw/fastf1_cache')
    seasons = [2022, 2023]  # 2024 not included as season hasn't started yet
    
    # Initialize collector
    collector = F1DataCollector(cache_dir, seasons)
    
    # Run collection
    collector.collect_all_seasons()

if __name__ == "__main__":
    main()
