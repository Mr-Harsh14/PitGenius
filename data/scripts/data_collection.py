"""
Data collection script using FastF1 to gather Formula 1 race data.
"""

import fastf1
from fastf1 import plotting
import os
from pathlib import Path
import pandas as pd

# Configure FastF1
fastf1.Cache.enable_cache('src/data/raw/cache')  # Cache directory
fastf1.plotting.setup_mpl(misc_mpl_mods=False)  # Disable future warning

def collect_historical_data(year: int = 2023):
    """
    Collect historical race data for the specified year.
    """
    # Get schedule for the year using Ergast API
    schedule = fastf1.get_event_schedule(year)
    
    for round_num in schedule.index:
        try:
            # Get race data for this round
            event = fastf1.get_event(year, round_num)
            session = event.get_race()
            session.load()
            
            # Save data to raw directory
            race_data_path = Path('src/data/raw/races') / str(event['EventName']).lower().replace(' ', '_')
            os.makedirs(race_data_path, exist_ok=True)
            
            # Save laps data
            session.laps.to_csv(race_data_path / 'laps.csv', index=False)
            
            # Save drivers data with full information
            driver_info = []
            for driver_number in session.drivers:
                driver_data = {
                    'DriverNumber': driver_number,
                    'BroadcastName': session.get_driver(driver_number)['BroadcastName'],
                    'FullName': session.get_driver(driver_number)['FullName'],
                    'Abbreviation': session.get_driver(driver_number)['Abbreviation'],
                    'TeamName': session.get_driver(driver_number)['TeamName'],
                    'TeamColor': session.get_driver(driver_number)['TeamColor']
                }
                driver_info.append(driver_data)
            
            drivers_df = pd.DataFrame(driver_info)
            drivers_df.to_csv(race_data_path / 'drivers.csv', index=False)
            
            # Save weather data if available
            if hasattr(session, 'weather_data'):
                session.weather_data.to_csv(race_data_path / 'weather.csv', index=False)
            
            print(f"Successfully collected data for {event['EventName']}")
            
        except Exception as e:
            print(f"Failed to collect data for Round {round_num}: {str(e)}")

if __name__ == "__main__":
    collect_historical_data()
