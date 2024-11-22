import fastf1
import requests
import pandas as pd
import os
import json
import logging
from typing import Dict, List, Any
from pathlib import Path

class F1DataCollector:
    def __init__(self):
        self.fastf1_session = None
        self.ergast_base_url = "http://ergast.com/api/f1"
        
        # Setup base directories
        self.base_path = Path('data')
        self.raw_data_path = self.base_path / 'raw'
        self.cache_path = self.base_path / 'cache'
        
        # Create all necessary directories
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep files
        (self.raw_data_path / '.gitkeep').touch()
        (self.cache_path / '.gitkeep').touch()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Enable FastF1 logging
        fastf1.set_log_level(logging.INFO)

    def initialize_session(self, year: int, gp: str, session: str):
        """Initialize FastF1 session"""
        self.logger.info(f"Initializing session for {year} {gp} {session}")
        
        # Enable cache with the correct path
        cache_dir = str(self.cache_path)
        self.logger.info(f"Setting cache directory to: {cache_dir}")
        fastf1.Cache.enable_cache(cache_dir)
        
        # Get and load session
        self.fastf1_session = fastf1.get_session(year, gp, session)
        self.logger.info("Loading session data...")
        self.fastf1_session.load()
        
        # Create session-specific directory
        session_name = f"{year}_{gp}_{session}".replace(" ", "_")
        self.session_path = self.raw_data_path / session_name
        self.session_path.mkdir(exist_ok=True)
        self.logger.info(f"Created session directory: {self.session_path}")

    def get_telemetry_data(self) -> Dict[str, Any]:
        """Collect and save telemetry data for all drivers"""
        if not self.fastf1_session:
            raise ValueError("Session not initialized")
        
        telemetry_data = {}
        
        # Get list of drivers
        drivers = self.fastf1_session.drivers
        print(f"Found drivers: {drivers}")
        
        for driver in drivers:
            print(f"Processing driver: {driver}")
            # Get car telemetry for each driver
            car_data = self.fastf1_session.laps.pick_driver(driver).get_car_data()
            if car_data is not None:
                # Save telemetry data
                filename = f"telemetry_{driver}.csv"
                save_path = self.session_path / filename
                print(f"Saving telemetry data to: {save_path}")
                car_data.to_csv(save_path)
                telemetry_data[driver] = car_data
            else:
                print(f"No telemetry data available for driver: {driver}")
        
        return telemetry_data

    def get_lap_times(self) -> pd.DataFrame:
        """Collect and save lap times data"""
        if not self.fastf1_session:
            raise ValueError("Session not initialized")
        
        # Get lap times for all drivers
        lap_times = self.fastf1_session.laps
        
        # Save lap times
        save_path = self.session_path / "lap_times.csv"
        print(f"Saving lap times to: {save_path}")
        lap_times.to_csv(save_path)
        
        return lap_times

    def get_position_data(self) -> Dict[str, pd.DataFrame]:
        """Collect and save position data for all drivers"""
        if not self.fastf1_session:
            raise ValueError("Session not initialized")
        
        position_data = {}
        drivers = self.fastf1_session.drivers
        
        for driver in drivers:
            # Get position data for each driver
            pos_data = self.fastf1_session.laps.pick_driver(driver).get_pos_data()
            if pos_data is not None:
                # Save position data
                filename = f"position_{driver}.csv"
                pos_data.to_csv(self.session_path / filename)
                position_data[driver] = pos_data
        
        return position_data

    def get_ergast_data(self, endpoint: str) -> Dict[str, Any]:
        """Retrieve and save data from Ergast API"""
        response = requests.get(f"{self.ergast_base_url}/{endpoint}.json")
        data = response.json()
        
        # Save Ergast data
        filename = f"ergast_{endpoint.replace('/', '_')}.json"
        with open(self.session_path / filename, 'w') as f:
            json.dump(data, f, indent=4)
        
        return data

    def collect_all_session_data(self):
        """Collect all available data for the current session"""
        if not self.fastf1_session:
            raise ValueError("Session not initialized")
            
        print("Collecting telemetry data...")
        self.get_telemetry_data()
        
        print("Collecting lap times...")
        self.get_lap_times()
        
        print("Collecting position data...")
        self.get_position_data()