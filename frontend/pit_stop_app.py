import streamlit as st
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import logging
from datetime import datetime
import requests
import json
from dotenv import load_dotenv

# Configure logging first, before anything else
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add project root to path
if os.environ.get('STREAMLIT_SHARING'):
    project_root = '/mount/src/pitgenius'
else:
    project_root = str(Path(__file__).parent.parent)

if project_root not in sys.path:
    sys.path.append(project_root)

# Import after adding project root to path
from src.models.train_random_forest import load_model
from src.models.predict_pit_stops import get_race_data, prepare_features, predict_race_pit_stops

# Initialize FastF1 cache
cache_dir = Path(project_root) / 'data' / 'raw' / 'fastf1_cache'
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

# Set up API configuration
RAPIDAPI_KEY = ''
RAPIDAPI_HOST = 'f1-motorsport-data.p.rapidapi.com'

# Try environment variables first
if 'RAPIDAPI_KEY' in os.environ:
    logger.info("Loading API key from environment variables")
    RAPIDAPI_KEY = os.environ['RAPIDAPI_KEY']
    RAPIDAPI_HOST = os.environ.get('RAPIDAPI_HOST', RAPIDAPI_HOST)
# Then try Streamlit secrets
elif hasattr(st, 'secrets'):
    logger.info("Checking for API key in Streamlit secrets")
    if 'RAPIDAPI_KEY' in st.secrets:
        logger.info("Loading API key from Streamlit secrets")
        RAPIDAPI_KEY = st.secrets['RAPIDAPI_KEY']
        if 'RAPIDAPI_HOST' in st.secrets:
            RAPIDAPI_HOST = st.secrets['RAPIDAPI_HOST']
    else:
        logger.warning("No RAPIDAPI_KEY found in Streamlit secrets")
else:
    logger.warning("No API keys found in environment or Streamlit secrets")

RAPIDAPI_BASE_URL = f"https://{RAPIDAPI_HOST}"

def rapidapi_request(endpoint, params=None):
    """Make a request to the RapidAPI F1 Motorsport Data API."""
    # Get the API key (try both module level and direct from secrets)
    api_key = RAPIDAPI_KEY
    api_host = RAPIDAPI_HOST
    
    # If no key, try to get directly from secrets (this works in other functions)
    if not api_key and hasattr(st, 'secrets'):
        logger.info("rapidapi_request: Trying to get API key directly from secrets")
        try:
            api_key = st.secrets.get('RAPIDAPI_KEY', '')
            if api_key:
                logger.info("rapidapi_request: Successfully loaded API key from secrets")
        except Exception as e:
            logger.warning(f"rapidapi_request: Error getting API key from secrets: {str(e)}")
    
    # Log access method for comparison with news function
    if api_key:
        masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
        logger.info(f"rapidapi_request using API key: {masked_key}")
    else:
        logger.warning("rapidapi_request: No API key available")
        return None
    
    if not api_key:
        logger.warning("No RapidAPI key configured. Set RAPIDAPI_KEY in .env file.")
        return None
        
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": api_host
    }
    
    url = f"https://{api_host}/{endpoint}"
    logger.info(f"Making API request to: {url}")
    
    try:
        response = requests.get(
            url,
            headers=headers,
            params=params
        )
        
        # Log the response status
        logger.info(f"API response status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"API returned non-200 status: {response.status_code}, Response: {response.text[:100]}...")
            return None
            
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Try to parse JSON
        try:
            data = response.json()
            # Log a sample of what we got back
            sample = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
            logger.info(f"API returned data: {sample}")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode API response: {e}")
            logger.error(f"Response content: {response.text[:100]}...")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        return None

def plot_driver_prediction(predictions: pd.DataFrame, driver_code: str):
    """Create a visualization of predicted pit stops for a driver."""
    # Filter predictions for this driver
    driver_predictions = predictions[
        predictions['Driver'] == driver_code
    ].sort_values('LapNumber')
    
    if driver_predictions.empty:
        st.warning(f"No predictions found for driver {driver_code}")
        return
    
    # Set style to default (light) theme
    plt.style.use('default')
    
    # Create figure with two subplots - even smaller size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 3), height_ratios=[2, 1], gridspec_kw={'hspace': 0.05})
    
    # Plot pit stop probability on top subplot
    ax1.plot(driver_predictions['LapNumber'], driver_predictions['PitProbability'], 
             label='Pit Stop Probability', color='#0066FF', alpha=0.8, linewidth=2)
    
    # Plot prediction threshold
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, 
                label='Prediction Threshold')
    
    # Plot predicted pit stops
    predicted_stops = driver_predictions[driver_predictions['PredictedPitStop']]['LapNumber']
    for pred_lap in predicted_stops:
        ax1.axvline(x=pred_lap, color='#00CC00', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add legend to top subplot with smaller font and better positioning
    ax1.legend(loc='upper right', framealpha=0.9, fontsize=8, bbox_to_anchor=(1.0, 0.95))
    
    # Customize top subplot
    ax1.set_title(f'Pit Stop Predictions - {driver_code}', pad=20)
    ax1.set_ylabel('Pit Stop Probability')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, max(driver_predictions['LapNumber']) + 1)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot predicted compounds on bottom subplot
    compounds = {
        'SOFT': '#FF1E1E',     # Bright red for soft
        'MEDIUM': '#FFF200',    # Bright yellow for medium
        'HARD': '#FFFFFF',      # White for hard
        'INTERMEDIATE': '#39B54A',  # Green for intermediate
        'WET': '#00A0DC'        # Blue for wet
    }
    
    # Set subplot background color
    ax2.set_facecolor('#1E1E1E')
    
    # Plot predicted compounds
    current_compound = None
    start_lap = 0
    
    sorted_laps = driver_predictions.sort_values('LapNumber')
    for i in range(len(sorted_laps)):
        lap = sorted_laps.iloc[i]
        if lap['CurrentCompound'] != current_compound or i == 0:
            if current_compound is not None and i > 0:
                end_lap = lap['LapNumber']
                ax2.axvspan(start_lap, end_lap, 
                           ymin=0.0, ymax=1.0,
                           color=compounds.get(current_compound, 'gray'), alpha=0.8)
                # Add compound label
                mid_lap = (start_lap + end_lap) / 2
                text_color = 'black' if current_compound in ['MEDIUM', 'HARD'] else 'white'
                ax2.text(mid_lap, 0.5, current_compound[0] if current_compound else '?', 
                        horizontalalignment='center', verticalalignment='center',
                        color=text_color, fontweight='bold')
            current_compound = lap['CurrentCompound']
            start_lap = lap['LapNumber']
    
    # Plot the last compound
    if current_compound is not None:
        ax2.axvspan(start_lap, max(sorted_laps['LapNumber']), 
                   ymin=0.0, ymax=1.0,
                   color=compounds.get(current_compound, 'gray'), alpha=0.8)
        mid_lap = (start_lap + max(sorted_laps['LapNumber'])) / 2
        text_color = 'black' if current_compound in ['MEDIUM', 'HARD'] else 'white'
        ax2.text(mid_lap, 0.5, current_compound[0] if current_compound else '?', 
                horizontalalignment='center', verticalalignment='center',
                color=text_color, fontweight='bold')
    
    # Plot predicted pit stops on compound subplot
    for pit_lap in predicted_stops:
        ax2.axvline(x=pit_lap, color='white', linestyle='--', alpha=0.8, linewidth=2)
    
    # Customize compound subplot
    ax2.set_xlabel('Lap Number')
    ax2.set_ylabel('Compound')
    ax2.set_yticks([])
    ax2.set_xlim(0, max(driver_predictions['LapNumber']) + 1)
    
    # Add compound legend with better formatting
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.8, label=compound)
                      for compound, color in compounds.items()]
    ax2.legend(handles=legend_elements, loc='upper right', ncol=3,
              fontsize=7, bbox_to_anchor=(1.0, 1.4),
              facecolor='#1E1E1E', edgecolor='gray')
    
    # Set figure background to white
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    
    # Adjust layout with minimal margins
    plt.tight_layout(pad=1.0)
    
    return fig

def get_historical_strategy(race_name: str, team: str, years: list = [2022, 2023]):
    """Get historical pit stop strategies for a team at a specific race."""
    strategies = []
    
    # Map current team names to historical names
    team_mapping = {
        'Red Bull Racing': ['Red Bull Racing', 'Red Bull'],
        'Mercedes': ['Mercedes'],
        'Ferrari': ['Ferrari'],
        'McLaren': ['McLaren'],
        'Aston Martin': ['Aston Martin', 'Racing Point'],
        'Alpine': ['Alpine', 'Renault'],
        'Williams': ['Williams'],
        'RB': ['AlphaTauri', 'Toro Rosso'],
        'Kick Sauber': ['Alfa Romeo', 'Sauber'],
        'Haas F1 Team': ['Haas F1 Team', 'Haas']
    }
    
    historical_team_names = team_mapping.get(team, [team])
    
    for year in years:
        try:
            # Get the race session
            session = fastf1.get_session(year, race_name, 'R')
            session.load()
            
            # Get team's drivers for this race
            team_drivers = []
            for driver_info in session.results.itertuples():
                if hasattr(driver_info, 'TeamName') and driver_info.TeamName in historical_team_names:
                    team_drivers.append(driver_info.DriverNumber)
            
            for driver_number in team_drivers:
                # Get driver's laps
                driver_laps = session.laps.pick_drivers(driver_number)
                
                if not driver_laps.empty:
                    # Get pit stops by looking at pit_in_time
                    pit_stops = driver_laps[~driver_laps['PitInTime'].isna()]
                    pit_laps = pit_stops['LapNumber'].tolist()
                    
                    # Get tire compounds for each stint
                    stints = []
                    current_compound = None
                    compounds = []
                    stint_start_lap = 1
                    
                    for _, lap in driver_laps.sort_values('LapNumber').iterrows():
                        if lap['Compound'] != current_compound:
                            if current_compound is not None:
                                compounds.append(current_compound)
                                stints.append({
                                    'start_lap': stint_start_lap,
                                    'end_lap': lap['LapNumber'] - 1,
                                    'compound': current_compound
                                })
                            current_compound = lap['Compound']
                            stint_start_lap = lap['LapNumber']
                    
                    # Add the last stint
                    if current_compound is not None:
                        compounds.append(current_compound)
                        stints.append({
                            'start_lap': stint_start_lap,
                            'end_lap': driver_laps['LapNumber'].max(),
                            'compound': current_compound
                        })
                    
                    # Get driver info and results
                    driver_results = session.results[session.results['DriverNumber'] == driver_number]
                    if not driver_results.empty:
                        driver_result = driver_results.iloc[0]
                        driver_code = driver_result.get('Abbreviation', str(driver_number))
                        result = driver_result['Position']
                        fastest_lap = driver_laps['LapTime'].min()
                        
                        strategies.append({
                            'Year': year,
                            'Driver': driver_code,
                            'Team': team,
                            'NumStops': len(pit_laps),
                            'PitLaps': pit_laps,
                            'Compounds': compounds,
                            'Stints': stints,
                            'Result': result,
                            'FastestLap': fastest_lap,
                            'RaceTime': driver_result.get('Time', None)
                        })
                    
        except Exception as e:
            logger.warning(f"Could not load data for {year} {race_name}: {str(e)}")
            continue
    
    return pd.DataFrame(strategies)

def plot_historical_strategies(strategies_df: pd.DataFrame, race_name: str, team: str):
    """Create a visualization of historical pit stop strategies."""
    if strategies_df.empty:
        st.warning(f"No historical data found for {team} at {race_name}")
        return
    
    # Set style to light theme
    plt.style.use('default')
    
    # Even smaller figure size
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, max(3, len(strategies_df))), 
                                  gridspec_kw={'height_ratios': [3, 1]}, layout='constrained')
    
    compounds_colors = {
        'SOFT': '#FF1E1E',     # Bright red for soft
        'MEDIUM': '#FFF200',    # Bright yellow for medium
        'HARD': '#808080',      # Gray for hard (better contrast on white)
        'INTERMEDIATE': '#39B54A',  # Green for intermediate
        'WET': '#00A0DC'        # Blue for wet
    }
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')
    
    y_positions = []
    y_labels = []
    
    # Plot strategy timeline
    for i, (_, strategy) in enumerate(strategies_df.iterrows()):
        y_pos = i * 2
        y_positions.append(y_pos)
        
        # Add driver name and result with color coding
        result_color = '#00FF00' if strategy['Result'] <= 3 else '#FFFFFF'
        y_labels.append(f"{strategy['Year']} - {strategy['Driver']} (P{strategy['Result']})")
        
        # Plot compounds
        for stint in strategy['Stints']:
            start = stint['start_lap']
            end = stint['end_lap']
            compound = stint['compound']
            
            # Plot stint bar
            ax1.barh(y_pos, end - start + 1, left=start, height=0.8,
                    color=compounds_colors.get(compound, 'gray'), alpha=0.8)
            
            # Add compound label with contrasting text color
            mid_point = start + (end - start) / 2
            text_color = 'black' if compound in ['MEDIUM', 'HARD'] else 'white'
            ax1.text(mid_point, y_pos, compound[0] if compound else '?',
                    ha='center', va='center', color=text_color, 
                    fontweight='bold', fontsize=10)
        
        # Plot pit stops with enhanced visibility
        for pit_lap in strategy['PitLaps']:
            # Add vertical line for pit stop
            ax1.axvline(x=pit_lap, ymin=(y_pos-0.4)/len(strategies_df)/2,
                       ymax=(y_pos+0.4)/len(strategies_df)/2,
                       color='black', linestyle='--', alpha=0.8, linewidth=2)
            
            # Add small marker at pit stop point
            ax1.plot([pit_lap], [y_pos], 'wo', markersize=6, alpha=0.8)
    
    # Customize strategy timeline with dark text
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(y_labels, color='black')
    ax1.set_xlabel('Lap Number', color='black', fontsize=10)
    ax1.set_title(f'Historical Pit Stop Strategies - {team} at {race_name}',
                  color='black', fontsize=12, pad=20)
    ax1.grid(True, alpha=0.2, color='gray')
    
    # Add compound legend with enhanced visibility and better positioning
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=compound)
                      for compound, color in compounds_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper right', ncol=3,
              fontsize=7, bbox_to_anchor=(1.0, 1.1),
              facecolor='white', edgecolor='gray')
    
    # Plot lap time comparison
    bar_width = 0.8
    for i, (_, strategy) in enumerate(strategies_df.iterrows()):
        if 'FastestLap' in strategy and pd.notnull(strategy['FastestLap']):
            lap_time_seconds = strategy['FastestLap'].total_seconds()
            # Plot bar with gradient alpha
            ax2.bar(i, lap_time_seconds, width=bar_width,
                   color=compounds_colors.get(strategy['Compounds'][-1], 'gray'),
                   alpha=0.8)
            # Add time label
            ax2.text(i, lap_time_seconds + 0.2, f"{lap_time_seconds:.1f}s",
                    ha='center', va='bottom', color='black', fontsize=9)
    
    # Customize lap time comparison with dark text
    ax2.set_xticks(range(len(strategies_df)))
    ax2.set_xticklabels([f"{s['Year']} - {s['Driver']}" for _, s in strategies_df.iterrows()],
                        rotation=45, ha='right', color='black')
    ax2.set_ylabel('Fastest Lap Time (s)', color='black', fontsize=10)
    ax2.set_title('Fastest Lap Comparison', color='black', fontsize=12, pad=20)
    ax2.grid(True, alpha=0.2, color='gray')
    
    # Set x-axis limits for strategy timeline
    ax1.set_xlim(0, 57)  # Bahrain GP is 57 laps
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def get_upcoming_event():
    """Get the next upcoming F1 event using RapidAPI."""
    try:
        # Try to get from RapidAPI
        if RAPIDAPI_KEY:
            # Get current season schedule
            schedule_data = rapidapi_request("season/schedule", {"year": 2025})
            
            if schedule_data and "events" in schedule_data:
                events = schedule_data["events"]
                
                # Find next race that hasn't happened yet
                today = datetime.now().date()
                
                for event in events:
                    event_date = datetime.strptime(event.get("date", ""), "%Y-%m-%d").date()
                    if event_date > today and event.get("type") == "Race":
                        circuit_name = event.get("circuit", {}).get("name", "TBA")
                        return {
                            "EventName": event.get("name", ""),
                            "EventDate": event_date,
                            "CircuitName": circuit_name,
                            "Location": f"{event.get('circuit', {}).get('location', {}).get('city', '')}, {event.get('circuit', {}).get('location', {}).get('country', '')}",
                            "RoundNumber": event.get("round", "")
                        }
        
        # Fall back to FastF1 if RapidAPI fails or is not configured
        today = datetime.now().date()
        schedule = fastf1.get_event_schedule(2025)
        
        # Convert EventDate to date object for comparison
        upcoming_events = schedule[schedule['EventDate'].dt.date > today].sort_values('EventDate')
        
        if not upcoming_events.empty:
            # Create a dictionary with field validation for the event
            event_data = upcoming_events.iloc[0]
            return {
                "EventName": event_data.get("EventName", ""),
                "EventDate": event_data.get("EventDate", ""),
                "CircuitName": event_data.get("CircuitName", "TBA"),
                "Location": event_data.get("Location", ""),
                "RoundNumber": event_data.get("RoundNumber", "")
            }
        else:
            return None
    except Exception as e:
        logger.error(f"Error getting upcoming event: {str(e)}")
        return None

def get_driver_standings():
    """Get current driver standings using RapidAPI."""
    try:
        # Try to get from RapidAPI
        if RAPIDAPI_KEY:
            standings_data = rapidapi_request("standings-drivers", {"year": 2025})
            
            if standings_data and "standings" in standings_data:
                standings = []
                
                # Debug: Log the first entry's structure to see where team info is
                if "entries" in standings_data["standings"] and standings_data["standings"]["entries"]:
                    first_entry = standings_data["standings"]["entries"][0]
                    logger.info(f"First entry structure: {json.dumps(first_entry, indent=2)[:500]}...")
                
                for entry in standings_data["standings"].get("entries", []):
                    if "athlete" in entry:
                        driver = entry.get("athlete", {})
                        
                        # Get team information - check all possible locations
                        team_name = ""
                        # Method 1: Check in the athlete object
                        if "team" in driver:
                            team_name = driver.get("team", {}).get("displayName", "")
                        # Method 2: Check directly in the entry
                        elif "team" in entry:
                            team_name = entry.get("team", {}).get("displayName", "")
                        # Method 3: Try to find it in stats
                        elif "stats" in entry:
                            for stat in entry["stats"]:
                                if stat.get("name", "").lower() in ["team", "constructor", "constructors"]:
                                    team_name = stat.get("displayValue", "")
                                    break
                        
                        # If still not found, look through the entire entry for any team reference
                        if not team_name:
                            # Convert to string and look for team indicators
                            entry_str = str(entry)
                            if "McLaren" in entry_str:
                                team_name = "McLaren"
                            elif "Red Bull" in entry_str:
                                team_name = "Red Bull"
                            elif "Ferrari" in entry_str:
                                team_name = "Ferrari"
                            elif "Mercedes" in entry_str:
                                team_name = "Mercedes"
                            elif "Williams" in entry_str:
                                team_name = "Williams"
                            elif "Aston Martin" in entry_str:
                                team_name = "Aston Martin"
                            elif "Haas" in entry_str:
                                team_name = "Haas"
                            elif "Alpine" in entry_str:
                                team_name = "Alpine"
                            elif "Sauber" in entry_str:
                                team_name = "Sauber"
                        
                        logger.info(f"Driver: {driver.get('abbreviation', '')}, Team found: {team_name}")
                        
                        standings.append({
                            "Position": entry.get("stats", [])[0].get("displayValue", "-") if entry.get("stats") else "-",
                            "Driver": driver.get("abbreviation", ""),
                            "Team": team_name,
                            "Points": entry.get("stats", [])[1].get("displayValue", "0") if len(entry.get("stats", [])) > 1 else "0"
                        })
                
                return pd.DataFrame(standings)
        
        # Fall back to FastF1 or placeholder data
        # Get 2025 schedule
        schedule = fastf1.get_event_schedule(2025)
        
        # Get the most recent completed race
        today = datetime.now().date()
        # Convert EventDate to date object for comparison
        completed_races = schedule[schedule['EventDate'].dt.date < today].sort_values('EventDate', ascending=False)
        
        if not completed_races.empty:
            latest_race = completed_races.iloc[0]
            latest_session = fastf1.get_session(2025, latest_race['EventName'], 'R')
            latest_session.load()
            
            # Create standings from results
            all_results = latest_session.results
            
            # Transform results into driver standings
            standings = []
            for _, driver in all_results.iterrows():
                standings.append({
                    'Position': driver['Position'] if 'Position' in driver else '-',
                    'Driver': driver['Abbreviation'] if 'Abbreviation' in driver else driver['DriverNumber'],
                    'Team': driver['TeamName'] if 'TeamName' in driver else '-',
                    'Points': driver['Points'] if 'Points' in driver else 0
                })
            
            return pd.DataFrame(standings)
        
        # Placeholder data if all methods fail
        return pd.DataFrame([
            {'Position': 1, 'Driver': 'VER', 'Team': 'Red Bull Racing', 'Points': 0},
            {'Position': 2, 'Driver': 'PER', 'Team': 'Red Bull Racing', 'Points': 0},
            {'Position': 3, 'Driver': 'HAM', 'Team': 'Mercedes', 'Points': 0}
        ])
    except Exception as e:
        logger.error(f"Error getting driver standings: {str(e)}")
        # Return placeholder data
        return pd.DataFrame([
            {'Position': 1, 'Driver': 'VER', 'Team': 'Red Bull Racing', 'Points': 0},
            {'Position': 2, 'Driver': 'PER', 'Team': 'Red Bull Racing', 'Points': 0},
            {'Position': 3, 'Driver': 'HAM', 'Team': 'Mercedes', 'Points': 0}
        ])

def get_team_standings():
    """Get current team standings using RapidAPI."""
    try:
        # Try to get from RapidAPI
        if RAPIDAPI_KEY:
            standings_data = rapidapi_request("standings-controllers", {"year": 2025})
            
            if standings_data and "standings" in standings_data:
                standings = []
                
                for entry in standings_data["standings"].get("entries", []):
                    if "team" in entry and "stats" in entry and len(entry["stats"]) >= 2:
                        team = entry.get("team", {})
                        # Get position from stats[0]
                        position = entry["stats"][0].get("displayValue", "-")
                        # Get points from stats[1]
                        points = entry["stats"][1].get("displayValue", "0")
                        
                        standings.append({
                            "Position": position,
                            "Team": team.get("displayName", ""),
                            "Points": points
                        })
                
                return pd.DataFrame(standings)
        
        # Fall back to calculating from driver standings
        driver_standings = get_driver_standings()
        
        if not driver_standings.empty:
            # Group by team and sum points
            team_standings = driver_standings.groupby('Team')['Points'].sum().reset_index()
            team_standings = team_standings.sort_values('Points', ascending=False)
            
            # Add position
            team_standings['Position'] = range(1, len(team_standings) + 1)
            
            # Reorder columns
            team_standings = team_standings[['Position', 'Team', 'Points']]
            
            return team_standings
        
        # Placeholder data if all methods fail
        return pd.DataFrame([
            {'Position': 1, 'Team': 'Red Bull Racing', 'Points': 0},
            {'Position': 2, 'Team': 'Mercedes', 'Points': 0},
            {'Position': 3, 'Team': 'Ferrari', 'Points': 0}
        ])
    except Exception as e:
        logger.error(f"Error getting team standings: {str(e)}")
        # Return placeholder data
        return pd.DataFrame([
            {'Position': 1, 'Team': 'Red Bull Racing', 'Points': 0},
            {'Position': 2, 'Team': 'Mercedes', 'Points': 0},
            {'Position': 3, 'Team': 'Ferrari', 'Points': 0}
        ])

def get_latest_news():
    """Get latest F1 news using RapidAPI."""
    try:
        # Get API key - first try module level, then try to get directly from secrets
        api_key = RAPIDAPI_KEY
        api_host = RAPIDAPI_HOST
        
        # Log the initial API key state
        logger.info(f"Initial API key state (empty = {api_key == ''})")
        
        # If no key yet, try to get directly from secrets (failsafe for deployed app)
        if not api_key and hasattr(st, 'secrets'):
            logger.info("News function: Trying to get API key directly from secrets")
            
            # First, let's log all available secret keys (safely)
            try:
                if hasattr(st.secrets, '_secrets'):
                    # Log the structure of the secrets (without showing actual values)
                    secret_keys = list(st.secrets._secrets.keys())
                    logger.info(f"Available secret keys: {secret_keys}")
                    
                    # Check for nested 'RAPIDAPI_KEY' 
                    if 'RAPIDAPI_KEY' in secret_keys:
                        logger.info("Found 'RAPIDAPI_KEY' in secret_keys")
                    # Check for different case variations
                    elif 'rapidapi_key' in secret_keys:
                        logger.info("Found 'rapidapi_key' in secret_keys (lowercase)")
                    elif 'RapidAPI_Key' in secret_keys:
                        logger.info("Found 'RapidAPI_Key' in secret_keys (mixed case)")
            except Exception as e:
                logger.warning(f"Error inspecting secrets structure: {str(e)}")
            
            # Try all possible formats and cases of the API key
            possible_keys = ['RAPIDAPI_KEY', 'rapidapi_key', 'RapidAPI_Key', 'RapidApiKey', 'rapid_api_key']
            for key_name in possible_keys:
                try:
                    logger.info(f"Trying to access secrets with key: {key_name}")
                    if key_name in st.secrets:
                        api_key = st.secrets[key_name]
                        logger.info(f"Found API key using key name: {key_name}")
                        break
                except Exception as e:
                    logger.warning(f"Error accessing secret with key {key_name}: {str(e)}")
            
            # Also check if there's a nested structure
            try:
                if hasattr(st.secrets, 'api') and hasattr(st.secrets.api, 'RAPIDAPI_KEY'):
                    api_key = st.secrets.api.RAPIDAPI_KEY
                    logger.info("Found API key in nested 'api.RAPIDAPI_KEY' structure")
            except Exception as e:
                logger.warning(f"Error checking nested secrets: {str(e)}")
            
            # If we found a key, look for the host as well
            if api_key:
                for host_name in ['RAPIDAPI_HOST', 'rapidapi_host', 'RapidAPI_Host']:
                    try:
                        if host_name in st.secrets:
                            api_host = st.secrets[host_name]
                            logger.info(f"Found API host using key name: {host_name}")
                            break
                    except Exception:
                        pass
        
        # Log API key status (masked for security)
        if api_key:
            masked_key = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "***"
            logger.info(f"News function using API key: {masked_key}")
        else:
            logger.warning("No API key available for news function")
        
        # Try to get from RapidAPI
        if api_key:
            # Try different possible news endpoint URLs
            urls_to_try = [
                f"https://{api_host}/articles",
                f"https://{api_host}/news",
                f"https://{api_host}/news-articles"
            ]
            
            for url in urls_to_try:
                logger.info(f"Trying news API URL: {url}")
                
                response = requests.get(
                    url,
                    headers={
                        "X-RapidAPI-Key": api_key,
                        "X-RapidAPI-Host": api_host
                    }
                )
                
                logger.info(f"News API response status code: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        news_data = response.json()
                        
                        # Try to determine the data structure
                        logger.info(f"Response structure: {type(news_data)}")
                        
                        # Try different response structures
                        articles_data = None
                        if isinstance(news_data, dict) and "articles" in news_data:
                            logger.info("Found 'articles' key in response")
                            articles_data = news_data["articles"]
                        elif isinstance(news_data, list):
                            logger.info("Found list of articles in response")
                            articles_data = news_data
                        
                        if articles_data and isinstance(articles_data, list):
                            logger.info(f"Successfully retrieved {len(articles_data)} news items")
                            # Create news items from returned data
                            news_items = []
                            for article in articles_data[:6]:  # Get top 6 news items
                                # Look for images using different possible path formats
                                image_url = None
                                if "images" in article and isinstance(article["images"], list) and article["images"]:
                                    for img in article["images"]:
                                        if "url" in img and img["url"].endswith((".jpg", ".jpeg", ".png")):
                                            image_url = img["url"]
                                            break
                                elif "image" in article and article["image"]:
                                    image_url = article["image"]
                                
                                # Get title, checking various possible field names
                                title = None
                                for field in ["headline", "title", "name", "displayName"]:
                                    if field in article and article[field]:
                                        title = article[field]
                                        break
                                
                                # Get description
                                snippet = None
                                for field in ["description", "snippet", "summary", "shortDescription"]:
                                    if field in article and article[field]:
                                        snippet = article[field]
                                        break
                                
                                # Get URL
                                url = None
                                for field in ["link", "url", "href"]:
                                    if field in article and article[field]:
                                        url = article[field]
                                        break
                                
                                if title and snippet:
                                    # Add news item
                                    news_items.append({
                                        "title": title,
                                        "date": "Latest",
                                        "snippet": snippet,
                                        "url": url if url else "#",
                                        "image_url": image_url
                                    })
                            
                            if news_items:
                                logger.info(f"Processed {len(news_items)} news items from {url}")
                                return news_items
                        else:
                            logger.warning(f"News API returned empty or invalid data format from {url}")
                    except Exception as e:
                        logger.error(f"Error processing data from {url}: {str(e)}")
                
                logger.warning("All news API URLs failed")
            
            # Get public fallback content
            try:
                # Try to get recent F1 news from a public source
                logger.info("Checking if F1 public news API is available as fallback")
                try:
                    # Try a simpler API first
                    logger.info("Trying Formula 1 news API")
                    f1_api_url = "https://ergast.com/api/f1/current/last/results.json"
                    response = requests.get(f1_api_url)
                    
                    if response.status_code == 200:
                        data = response.json()
                        race_info = data.get('MRData', {}).get('RaceTable', {}).get('Races', [{}])[0]
                        
                        if race_info:
                            race_name = race_info.get('raceName', 'Formula 1 Race')
                            circuit = race_info.get('Circuit', {}).get('circuitName', 'Unknown Circuit')
                            date = race_info.get('date', 'Recent')
                            
                            results = race_info.get('Results', [])
                            news_items = []
                            
                            # Create news items from race results
                            if results:
                                winner = results[0].get('Driver', {})
                                winner_name = f"{winner.get('givenName', '')} {winner.get('familyName', '')}"
                                
                                news_items.append({
                                    "title": f"{winner_name} Wins {race_name}",
                                    "date": date,
                                    "snippet": f"{winner_name} secured victory at the {circuit}. Check the full race results.",
                                    "url": "https://www.formula1.com/en/results.html",
                                    "image_url": None
                                })
                                
                                # Add podium finishers
                                for i, result in enumerate(results[1:3], 2):
                                    driver = result.get('Driver', {})
                                    driver_name = f"{driver.get('givenName', '')} {driver.get('familyName', '')}"
                                    
                                    news_items.append({
                                        "title": f"{driver_name} Finishes P{i} at {race_name}",
                                        "date": date,
                                        "snippet": f"{driver_name} secured a P{i} finish at the {circuit}.",
                                        "url": "https://www.formula1.com/en/results.html",
                                        "image_url": None
                                    })
                                
                                return news_items
                except Exception as e:
                    logger.error(f"Error with F1 public API fallback: {str(e)}")
                
                # Use placeholder news as final fallback
                public_news_url = "https://www.formula1.com/en/latest/all.html"
                logger.info(f"Using placeholder news with link to: {public_news_url}")
                
                return [
                    {"title": "F1 News [Fallback]", "date": "Latest", 
                     "snippet": "API connection failed. Showing backup content.", "url": "https://www.formula1.com/en/latest/all.html", "image_url": None},
                    {"title": "Visit Formula1.com", "date": "Latest", 
                     "snippet": "Check formula1.com for the latest news and updates.", "url": "https://www.formula1.com", "image_url": None},
                    {"title": "API Troubleshooting", "date": "Latest", 
                     "snippet": "If you're seeing this message, check your RapidAPI key configuration.", "url": "#", "image_url": None}
                ]
            except Exception as e:
                logger.error(f"Error with all news fallbacks: {str(e)}")
        
        # Final fallback - hardcoded news items
        logger.info("Using hardcoded news items as final fallback")
        return [
            {"title": "Red Bull Dominates Testing", "date": "March 2024", "snippet": "Red Bull shows impressive pace in pre-season testing.", "url": "#", "image_url": None},
            {"title": "Ferrari Unveils New Upgrades", "date": "March 2024", "snippet": "Ferrari brings significant aerodynamic package to next race.", "url": "#", "image_url": None},
            {"title": "Mercedes Addresses Porpoising Issues", "date": "February 2024", "snippet": "Team reports progress on fixing persistent bouncing problems.", "url": "#", "image_url": None}
        ]
    except Exception as e:
        logger.error(f"Error getting latest news: {str(e)}")
        return [
            {"title": "Red Bull Dominates Testing", "date": "March 2024", "snippet": "Red Bull shows impressive pace in pre-season testing.", "url": "#", "image_url": None},
            {"title": "Ferrari Unveils New Upgrades", "date": "March 2024", "snippet": "Ferrari brings significant aerodynamic package to next race.", "url": "#", "image_url": None},
            {"title": "Mercedes Addresses Porpoising Issues", "date": "February 2024", "snippet": "Team reports progress on fixing persistent bouncing problems.", "url": "#", "image_url": None}
        ]

def get_full_season_schedule(year=2025):
    """Get the full F1 season schedule using RapidAPI."""
    try:
        # Try to get from RapidAPI
        if RAPIDAPI_KEY:
            # Get current season schedule
            schedule_data = rapidapi_request("season", {"year": year})
            
            if schedule_data and "events" in schedule_data:
                events = []
                today = datetime.now().date()
                
                for event in schedule_data["events"]:
                    # Parse date
                    date_str = event.get("date", "")
                    try:
                        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                        formatted_date = event_date.strftime("%d %b %Y")
                    except ValueError:
                        formatted_date = date_str
                        event_date = datetime.now().date()  # Fallback for status calculation
                    
                    circuit_name = event.get("circuit", {}).get("name", "")
                    location = event.get("circuit", {}).get("location", {})
                    city = location.get("city", "")
                    country = location.get("country", "")
                    
                    events.append({
                        "Round": event.get("round", ""),
                        "Name": event.get("name", ""),
                        "Circuit": circuit_name,
                        "Location": f"{city}, {country}" if city and country else "",
                        "Date": formatted_date,
                        "Status": "Completed" if event_date < today else "Upcoming"
                    })
                
                return pd.DataFrame(events)
        
        # Fall back to FastF1 if RapidAPI fails or is not configured
        schedule = fastf1.get_event_schedule(year)
        
        # Convert to our format
        events = []
        today = datetime.now().date()
        
        for _, event in schedule.iterrows():
            if event['EventFormat'] == 'conventional':
                # Convert pandas timestamp to date for comparison
                event_date = event['EventDate'].to_pydatetime().date()
                
                events.append({
                    "Round": event['RoundNumber'],
                    "Name": event['EventName'],
                    "Circuit": event['EventName'],  # Use EventName as fallback for CircuitName
                    "Location": event['Location'],
                    "Date": event_date.strftime("%d %b %Y"),
                    "Status": "Completed" if event_date < today else "Upcoming"
                })
        
        return pd.DataFrame(events)
    except Exception as e:
        logger.error(f"Error getting season schedule: {str(e)}")
        return pd.DataFrame()

def create_dashboard():
    """Create the main dashboard with standings, schedule, and news."""
    st.header("📊 F1 Dashboard")
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .dashboard-card {
        border-radius: 10px;
        border: 1px solid #e6e6e6;
        padding: 15px;
        margin-bottom: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .card-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333333;
        border-bottom: 2px solid #ff1e1e;
        padding-bottom: 8px;
    }
    .news-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: white;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .news-date {
        color: #777;
        font-size: 0.8em;
        margin-bottom: 10px;
    }
    .news-title {
        color: #333;
        margin-top: 0;
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 10px;
    }
    .news-snippet {
        color: #444;
        margin-bottom: 15px;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .news-link {
        color: #ff1e1e;
        text-decoration: none;
        font-weight: 500;
    }
    .red-button {
        background-color: #ff1e1e;
        color: white; 
        border: none;
        border-radius: 5px;
        padding: 10px 15px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # F1 Team colors for reference
    team_colors = {
        "Red Bull Racing": "#0600EF",
        "Mercedes": "#00D2BE",
        "Ferrari": "#DC0000",
        "McLaren": "#FF8700",
        "Aston Martin": "#006F62",
        "Alpine": "#0090FF",
        "Williams": "#005AFF",
        "RB": "#1E41FF",
        "Kick Sauber": "#900000",
        "Haas F1 Team": "#FFFFFF",
        "Racing Bulls": "#1E41FF",
        "Racing Point": "#F596C8",
        "Renault": "#FFF500",
        "Alfa Romeo": "#900000",
        "Toro Rosso": "#469BFF",
        "AlphaTauri": "#2B4562",
    }
    
    # Add Full Season Schedule Section at the top
    with st.container():
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="card-title">🗓️ 2025 F1 Season Schedule</h2>', unsafe_allow_html=True)
        
        # Get full season schedule
        schedule = get_full_season_schedule(2025)
        
        if not schedule.empty:
            # Add season progress
            completed_races = len(schedule[schedule['Status'] == 'Completed'])
            total_races = len(schedule)
            progress_percentage = (completed_races / total_races) * 100 if total_races > 0 else 0
            
            st.markdown(f"**Season Progress:** {completed_races} of {total_races} races completed")
            st.progress(progress_percentage / 100)
            
            # Show all races in a single view
            st.dataframe(
                schedule,
                column_config={
                    "Round": st.column_config.NumberColumn("Round", help="Race round number", format="%d"),
                    "Name": st.column_config.TextColumn("Grand Prix", width="medium"),
                    "Circuit": st.column_config.TextColumn("Circuit", width="medium"),
                    "Location": st.column_config.TextColumn("Location", width="medium"),
                    "Date": st.column_config.TextColumn("Date"),
                    "Status": st.column_config.TextColumn("Status", width="small")
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Season schedule not available.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create two columns for standings
    col1, col2 = st.columns(2)
    
    # Driver Standings Card
    with col1:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="card-title">🏆 Driver Standings</h2>', unsafe_allow_html=True)
        
        driver_standings = get_driver_standings()
        
        if not driver_standings.empty:
            # Limit to top 5 drivers
            top_drivers = driver_standings.head(5).copy()
            
            # Add team color indicators to the dataframe
            # Format points as integers
            top_drivers_display = top_drivers.copy()
            
            # Display using Streamlit's dataframe for consistency
            st.dataframe(
                top_drivers_display,
                column_config={
                    "Position": st.column_config.NumberColumn("Pos", format="%d"),
                    "Driver": st.column_config.TextColumn("Driver"),
                    "Team": st.column_config.TextColumn("Team"),
                    "Points": st.column_config.NumberColumn("Points", format="%d")
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Add view more button
            if st.button("View All Drivers", key="view_all_drivers", use_container_width=True, type="primary"):
                all_drivers_display = driver_standings.copy()
                st.dataframe(
                    all_drivers_display,
                    column_config={
                        "Position": st.column_config.NumberColumn("Pos", format="%d"),
                        "Driver": st.column_config.TextColumn("Driver"),
                        "Team": st.column_config.TextColumn("Team"),
                        "Points": st.column_config.NumberColumn("Points", format="%d")
                    },
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.info("Driver standings not available.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Team Standings Card
    with col2:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="card-title">🏭 Team Standings</h2>', unsafe_allow_html=True)
        
        team_standings = get_team_standings()
        
        if not team_standings.empty:
            team_standings_display = team_standings.copy()
            
            # Display using Streamlit's dataframe for consistency
            st.dataframe(
                team_standings_display,
                column_config={
                    "Position": st.column_config.NumberColumn("Pos", format="%d"),
                    "Team": st.column_config.TextColumn("Team"),
                    "Points": st.column_config.NumberColumn("Points", format="%d")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Team standings not available.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Latest News Card (full width)
    with st.container():
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="card-title">📰 Latest F1 News</h2>', unsafe_allow_html=True)
        
        # Get news items
        news_items = get_latest_news()
        
        # Create columns for news items
        num_news_items = min(len(news_items), 3)
        news_cols = st.columns(num_news_items)
        
        # Display news items in columns
        for idx, item in enumerate(news_items[:num_news_items]):
            with news_cols[idx]:
                # Add image if available
                if item.get('image_url'):
                    st.image(item['image_url'], use_column_width=True)
                
                st.markdown(f"""
                <div class="news-card">
                    <h4 class="news-title">{item['title']}</h4>
                    <p class="news-date">{item['date']}</p>
                    <p class="news-snippet">{item['snippet']}</p>
                    <a href="{item['url']}" target="_blank" class="news-link">Read more →</a>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Strategy Preview section
    with st.container():
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown('<h2 class="card-title">🔍 Strategy Predictions</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        Get a sneak peek at predicted pit stop strategies for the upcoming race. 
        Use the <strong>Predictions</strong> tab in the navigation menu for detailed analysis.
        """, unsafe_allow_html=True)
        
        # Preview button that redirects to predictions tab
        if st.button("View Strategy Predictions", key="view_predictions", use_container_width=True, type="primary"):
            st.session_state.app_mode = "🔮 Predictions"
            st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="PitGenius - F1 Pit Stop Predictions", layout="wide")
    
    # Title and description in the header
    st.title("🏎️ PitGenius: F1 Pit Stop Predictions")
    
    # Initialize session state for navigation
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "📊 Dashboard"
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Navigation Options",  # Added label for accessibility
        ["📊 Dashboard", "🔮 Predictions", "📈 Historical Analysis"],
        index=["📊 Dashboard", "🔮 Predictions", "📈 Historical Analysis"].index(st.session_state.app_mode),
        label_visibility="collapsed"  # Hide the label but keep it for accessibility
    )
    
    # Update session state
    st.session_state.app_mode = app_mode
    
    # Sidebar info/branding
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "PitGenius uses machine learning to predict F1 pit stop strategies "
        "based on historical data from the 2022-2023 seasons."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Developed with")
    st.sidebar.markdown("- FastF1 🏎️")
    st.sidebar.markdown("- Streamlit 📊")
    st.sidebar.markdown("- Scikit-learn 🤖")
    
    # Add API status indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### API Status")
    if RAPIDAPI_KEY:
        # Test API connection
        if st.sidebar.button("Test API Connection"):
            with st.sidebar:
                with st.spinner("Testing API connection..."):
                    # Show environment info for debugging
                    if st.checkbox("Show environment details"):
                        st.write("Running in Streamlit Cloud:" if 'STREAMLIT_SHARING' in os.environ else "Running locally")
                        st.write(f"API key source: {'Streamlit Secrets' if 'RAPIDAPI_KEY' not in os.environ else 'Environment Variable'}")
                        
                    # Test the connection
                    test_response = rapidapi_request("news")
                    if test_response:
                        st.success("✅ RapidAPI Connection Successful")
                        st.write(f"Retrieved {len(test_response)} news items")
                    else:
                        st.error("❌ RapidAPI Connection Failed")
                        # Show more diagnostic information
                        st.info("Check API key and rate limits")
                        
                        # Test with basic request
                        try:
                            simple_response = requests.get(
                                f"{RAPIDAPI_BASE_URL}/seasons",
                                headers={
                                    "X-RapidAPI-Key": RAPIDAPI_KEY,
                                    "X-RapidAPI-Host": RAPIDAPI_HOST
                                }
                            )
                            st.write(f"Status code: {simple_response.status_code}")
                            if simple_response.status_code != 200:
                                st.write(f"Error response: {simple_response.text[:200]}...")
                        except Exception as e:
                            st.write(f"Request error: {str(e)}")
                            
        st.sidebar.success("✅ RapidAPI Key Configured")
        # Show masked API key
        masked_key = RAPIDAPI_KEY[:4] + "..." + RAPIDAPI_KEY[-4:] if len(RAPIDAPI_KEY) > 8 else "***"
        st.sidebar.code(f"API Key: {masked_key}", language=None)
    else:
        st.sidebar.warning("⚠️ RapidAPI Not Configured")
        st.sidebar.info("To configure the RapidAPI key:")
        st.sidebar.code("""
# Create a .env file in project root
# Add these lines:
RAPIDAPI_KEY=your_api_key_here
RAPIDAPI_HOST=f1-motorsport-data.p.rapidapi.com
        """)
        with st.sidebar.expander("How to get an API key"):
            st.markdown("""
            1. Visit [RapidAPI F1 Motorsport Data](https://rapidapi.com/sportcontentapi/api/f1-motorsport-data/)
            2. Sign up for a RapidAPI account
            3. Subscribe to the API (there is a free tier)
            4. Copy your API key from the dashboard
            5. Add it to your .env file
            """)
    
    # Get available races for 2025
    schedule = fastf1.get_event_schedule(2025)
    races = schedule[schedule['EventFormat'] == 'conventional']['EventName'].tolist()
    
    # Show content based on selected tab
    if app_mode == "📊 Dashboard":
        create_dashboard()
    else:
        # For predictions and historical analysis modes, show selection controls at top
        # Create 3 columns for selections
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Race selection
            selected_race = st.selectbox(
                "Select Race",
                races
            )
        
        # Predefined list of 2024 F1 drivers
        drivers_2024 = [
            {'code': 'VER', 'name': 'Max Verstappen', 'team': 'Red Bull Racing'},
            {'code': 'PER', 'name': 'Sergio Perez', 'team': 'Red Bull Racing'},
            {'code': 'HAM', 'name': 'Lewis Hamilton', 'team': 'Mercedes'},
            {'code': 'RUS', 'name': 'George Russell', 'team': 'Mercedes'},
            {'code': 'LEC', 'name': 'Charles Leclerc', 'team': 'Ferrari'},
            {'code': 'SAI', 'name': 'Carlos Sainz', 'team': 'Ferrari'},
            {'code': 'NOR', 'name': 'Lando Norris', 'team': 'McLaren'},
            {'code': 'PIA', 'name': 'Oscar Piastri', 'team': 'McLaren'},
            {'code': 'ALO', 'name': 'Fernando Alonso', 'team': 'Aston Martin'},
            {'code': 'STR', 'name': 'Lance Stroll', 'team': 'Aston Martin'},
            {'code': 'GAS', 'name': 'Pierre Gasly', 'team': 'Alpine'},
            {'code': 'OCO', 'name': 'Esteban Ocon', 'team': 'Alpine'},
            {'code': 'ALB', 'name': 'Alexander Albon', 'team': 'Williams'},
            {'code': 'SAR', 'name': 'Logan Sargeant', 'team': 'Williams'},
            {'code': 'BOT', 'name': 'Valtteri Bottas', 'team': 'Kick Sauber'},
            {'code': 'ZHO', 'name': 'Guanyu Zhou', 'team': 'Kick Sauber'},
            {'code': 'RIC', 'name': 'Daniel Ricciardo', 'team': 'RB'},
            {'code': 'TSU', 'name': 'Yuki Tsunoda', 'team': 'RB'},
            {'code': 'MAG', 'name': 'Kevin Magnussen', 'team': 'Haas F1 Team'},
            {'code': 'HUL', 'name': 'Nico Hulkenberg', 'team': 'Haas F1 Team'}
        ]
        
        # Group drivers by team
        teams = {}
        for driver in drivers_2024:
            if driver['team'] not in teams:
                teams[driver['team']] = []
            teams[driver['team']].append(driver)
        
        with col2:
            # Create team selection
            selected_team = st.selectbox(
                "Select Team",
                options=list(teams.keys())
            )
        
        # Then filter drivers by selected team
        team_drivers = teams[selected_team]
        
        with col3:
            selected_driver = st.selectbox(
                "Select Driver",
                options=[d['code'] for d in team_drivers],
                format_func=lambda x: next(d['name'] for d in team_drivers if d['code'] == x)
            )
        
        # Add some spacing
        st.markdown("---")
        
        # Show appropriate content based on the selected mode
        if app_mode == "🔮 Predictions":
            try:
                # Load model and make predictions
                with st.spinner("Making predictions..."):
                    # Load trained model
                    model = load_model([2022, 2023])
                    
                    # Get race data
                    race_data = get_race_data(2024, selected_race)
                    
                    # Prepare features
                    features = prepare_features(race_data)
                    
                    # Make predictions
                    predictions = predict_race_pit_stops(model, features)
                    
                    if not predictions.empty:
                        # Ensure current compound is properly set for visualization
                        compound_map = {
                            1: 'SOFT',
                            2: 'MEDIUM', 
                            3: 'HARD',
                            4: 'INTERMEDIATE',
                            5: 'WET'
                        }
                        
                        # If CurrentCompound is missing or None, set default compound based on index
                        if 'CurrentCompound' not in predictions.columns or predictions['CurrentCompound'].isna().any():
                            # Set a default compound (MEDIUM)
                            predictions['CurrentCompound'] = 'MEDIUM'
                            
                            # For each driver, set a consistent compound
                            for driver in predictions['Driver'].unique():
                                driver_idx = list(predictions['Driver'].unique()).index(driver) % 3
                                if driver_idx == 0:
                                    predictions.loc[predictions['Driver'] == driver, 'CurrentCompound'] = 'SOFT'
                                elif driver_idx == 1:
                                    predictions.loc[predictions['Driver'] == driver, 'CurrentCompound'] = 'MEDIUM'
                                else:
                                    predictions.loc[predictions['Driver'] == driver, 'CurrentCompound'] = 'HARD'
                        
                        # Create two columns with different widths and add max-width constraint
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Add a container with max width
                            with st.container():
                                st.markdown(
                                    """
                                    <style>
                                    .plot-container {
                                        max-width: 800px;
                                        margin: auto;
                                    }
                                    </style>
                                    """,
                                    unsafe_allow_html=True
                                )
                                # Create visualization
                                fig = plot_driver_prediction(predictions, selected_driver)
                                st.pyplot(fig, use_container_width=True)
                        
                        with col2:
                            # Display strategy summary
                            st.subheader("Predicted Strategy Summary")
                            driver_preds = predictions[
                                (predictions['Driver'] == selected_driver) & 
                                predictions['PredictedPitStop']
                            ]
                            
                            if not driver_preds.empty:
                                st.write(f"Number of predicted pit stops: {len(driver_preds)}")
                                
                                # Create strategy table
                                strategy_data = []
                                prev_compound = predictions[predictions['Driver'] == selected_driver].iloc[0]['CurrentCompound']
                                
                                for i, (_, stop) in enumerate(driver_preds.iterrows(), 1):
                                    # Find next compound after this pit stop
                                    next_lap_data = predictions[
                                        (predictions['Driver'] == selected_driver) & 
                                        (predictions['LapNumber'] > stop['LapNumber'])
                                    ]
                                    
                                    if next_lap_data.empty:
                                        next_compound = "UNKNOWN"
                                    else:
                                        next_compound = next_lap_data.iloc[0]['CurrentCompound']
                                    
                                    strategy_data.append({
                                        'Stop': f"Pit Stop {i}",
                                        'Lap': int(stop['LapNumber']),
                                        'From': prev_compound,
                                        'To': next_compound,
                                        'Probability': f"{stop['PitProbability']:.2%}"
                                    })
                                    prev_compound = next_compound
                                
                                if strategy_data:
                                    st.table(pd.DataFrame(strategy_data))
                            else:
                                st.write("No pit stops predicted for this driver.")
                    else:
                        st.error("Error making predictions. Please try again.")
                        
            except Exception as e:
                st.error(f"Error loading race data: {str(e)}")
                logger.error(f"Error in Streamlit app: {str(e)}")
        
        elif app_mode == "📈 Historical Analysis":
            try:
                with st.spinner("Loading historical data..."):
                    # Get historical strategies
                    historical_data = get_historical_strategy(selected_race, selected_team)
                    
                    if not historical_data.empty:
                        # Create two columns with max-width constraint
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Add a container with max width
                            with st.container():
                                st.markdown(
                                    """
                                    <style>
                                    .plot-container {
                                        max-width: 800px;
                                        margin: auto;
                                    }
                                    </style>
                                    """,
                                    unsafe_allow_html=True
                                )
                                # Plot historical strategies
                                fig = plot_historical_strategies(historical_data, selected_race, selected_team)
                                st.pyplot(fig, use_container_width=True)
                        
                        with col2:
                            # Display summary statistics
                            st.subheader("Historical Strategy Summary")
                            
                            avg_stops = historical_data['NumStops'].mean()
                            st.write(f"Average number of pit stops: {avg_stops:.1f}")
                            
                            # Most common compounds
                            all_compounds = [compound for compounds in historical_data['Compounds'] for compound in compounds]
                            if all_compounds:
                                compound_counts = pd.Series(all_compounds).value_counts()
                                st.write("Most used tire compounds:")
                                for compound, count in compound_counts.items():
                                    st.write(f"- {compound}: {count} times")
                            
                            # Display detailed data
                            st.subheader("Detailed Historical Data")
                            display_data = historical_data[['Year', 'Driver', 'NumStops', 'Result']].copy()
                            display_data = display_data.sort_values(['Year', 'Result'])
                            st.dataframe(display_data)
                    else:
                        st.warning(f"No historical data found for {selected_team} at {selected_race}")
                    
            except Exception as e:
                st.error(f"Error loading historical data: {str(e)}")
                logger.error(f"Error loading historical data: {str(e)}")

if __name__ == "__main__":
    main() 