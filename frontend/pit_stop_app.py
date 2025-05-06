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
import random

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
    
    # Create figure with two subplots - larger size for better visibility
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[2, 1], gridspec_kw={'hspace': 0.15})
    
    # Plot pit stop probability on top subplot with better styling
    ax1.plot(driver_predictions['LapNumber'], driver_predictions['PitProbability'], 
             label='Pit Stop Probability', color='#0066FF', alpha=0.9, linewidth=2.5)
    
    # Plot prediction threshold with clearer style
    ax1.axhline(y=0.5, color='#888888', linestyle='--', alpha=0.7, 
                label='Prediction Threshold')
    
    # Plot predicted pit stops with more visible lines
    predicted_stops = driver_predictions[driver_predictions['PredictedPitStop']]['LapNumber']
    for pred_lap in predicted_stops:
        ax1.axvline(x=pred_lap, color='#00CC00', linestyle='-', alpha=0.7, linewidth=1.5)
        # Add a highlight marker at the peak for better visibility
        prob_at_stop = driver_predictions[driver_predictions['LapNumber'] == pred_lap]['PitProbability'].values[0]
        ax1.plot(pred_lap, prob_at_stop, 'o', color='#00CC00', markersize=8, alpha=0.8)
    
    # Move legend outside the plot to avoid overlap
    ax1.legend(loc='upper left', framealpha=0.9, fontsize=10, 
              bbox_to_anchor=(0, -0.05), ncol=2)
    
    # Customize top subplot with clearer styling
    ax1.set_title(f'Pit Stop Predictions - {driver_code}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pit Stop Probability', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(driver_predictions['LapNumber']) + 1)
    ax1.set_ylim(-0.05, 1.05)
    
    # Add more visible x-axis gridlines
    ax1.set_xticks(range(0, int(max(driver_predictions['LapNumber'])) + 5, 5))
    
    # Set y-axis ticks
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Plot predicted compounds on bottom subplot
    compounds = {
        'SOFT': '#FF1E1E',     # Bright red for soft
        'MEDIUM': '#FFF200',    # Bright yellow for medium
        'HARD': '#FFFFFF',      # White for hard
        'INTERMEDIATE': '#39B54A',  # Green for intermediate
        'WET': '#00A0DC'        # Blue for wet
    }
    
    # Set subplot background color - darker for better contrast
    ax2.set_facecolor('#1E1E1E')
    
    # Plot predicted compounds with better styling
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
                # Add larger, clearer compound label
                mid_lap = (start_lap + end_lap) / 2
                text_color = 'black' if current_compound in ['MEDIUM', 'HARD'] else 'white'
                ax2.text(mid_lap, 0.5, current_compound[0] if current_compound else '?', 
                        horizontalalignment='center', verticalalignment='center',
                        color=text_color, fontweight='bold', fontsize=14)
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
                color=text_color, fontweight='bold', fontsize=14)
    
    # Plot predicted pit stops on compound subplot with clearer lines
    for pit_lap in predicted_stops:
        ax2.axvline(x=pit_lap, color='white', linestyle='-', alpha=0.9, linewidth=1.5)
    
    # Customize compound subplot with better styling
    ax2.set_xlabel('Lap Number', fontsize=12)
    ax2.set_ylabel('Compound', fontsize=12)
    ax2.set_yticks([])
    ax2.set_xlim(0, max(driver_predictions['LapNumber']) + 1)
    
    # Use same x-axis ticks as top plot for consistency
    ax2.set_xticks(range(0, int(max(driver_predictions['LapNumber'])) + 5, 5))
    
    # Move compound legend outside the plot for clarity
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.8, label=compound)
                      for compound, color in compounds.items()]
    ax2.legend(handles=legend_elements, loc='lower right', ncol=5,
              fontsize=9, bbox_to_anchor=(1.0, -0.3),
              facecolor='white', edgecolor='gray')
    
    # Set figure background to white
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    
    # Add a box around the plots for better definition
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_color('#cccccc')
        ax.spines['right'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        ax.spines['left'].set_color('#cccccc')
    
    # Adjust layout with better spacing
    plt.tight_layout(pad=2.0)
    
    return fig

def get_historical_strategy(race_name: str, team: str, years: list = [2022, 2023, 2024]):
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
                    # Ensure lap numbers are integers for pit stops
                    pit_laps = [int(lap) for lap in pit_stops['LapNumber'].tolist()]
                    
                    # Get tire compounds for each stint
                    stints = []
                    current_compound = None
                    compounds = []
                    stint_start_lap = 1
                    
                    for _, lap in driver_laps.sort_values('LapNumber').iterrows():
                        lap_number = int(lap['LapNumber'])  # Convert to integer
                        if lap['Compound'] != current_compound:
                            if current_compound is not None:
                                compounds.append(current_compound)
                                stints.append({
                                    'start_lap': int(stint_start_lap),  # Ensure integer
                                    'end_lap': lap_number - 1,  # Already an integer
                                    'compound': current_compound
                                })
                            current_compound = lap['Compound']
                            stint_start_lap = lap_number  # Store as integer
                    
                    # Add the last stint
                    if current_compound is not None:
                        compounds.append(current_compound)
                        stints.append({
                            'start_lap': int(stint_start_lap),  # Ensure integer
                            'end_lap': int(driver_laps['LapNumber'].max()),  # Ensure integer
                            'compound': current_compound
                        })
                    
                    # Get driver info and results
                    driver_results = session.results[session.results['DriverNumber'] == driver_number]
                    if not driver_results.empty:
                        driver_result = driver_results.iloc[0]
                        driver_code = driver_result.get('Abbreviation', str(driver_number))
                        result = int(driver_result['Position'])  # Ensure integer
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
            logger.error(f"Error loading historical data: {str(e)}")
            continue
    
    return pd.DataFrame(strategies)

def plot_historical_strategies(strategies_df: pd.DataFrame, race_name: str, team: str):
    """Create a visualization of historical pit stop strategies."""
    if strategies_df.empty:
        st.warning(f"No historical data found for {team} at {race_name}")
        return
    
    # Set style to light theme
    plt.style.use('default')
    
    # Larger figure size for better visibility and prevent overlapping
    # Increase height based on number of strategies and ensure minimum sizes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, max(8, len(strategies_df) * 1.2)), 
                                  gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.5})
    
    compounds_colors = {
        'SOFT': '#FF1E1E',      # Bright red for soft
        'MEDIUM': '#FFF200',    # Bright yellow for medium
        'HARD': '#FFFFFF',      # White for hard
        'INTERMEDIATE': '#39B54A',  # Green for intermediate
        'WET': '#00A0DC'        # Blue for wet
    }
    
    # Set white background with light grid
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('#f8f8f8')
    ax2.set_facecolor('#f8f8f8')
    
    # Add border
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_color('#dddddd')
        ax.spines['right'].set_color('#dddddd')
        ax.spines['bottom'].set_color('#dddddd')
        ax.spines['left'].set_color('#dddddd')
    
    y_positions = []
    y_labels = []
    
    # Plot strategy timeline
    for i, (_, strategy) in enumerate(strategies_df.iterrows()):
        y_pos = i * 2
        y_positions.append(y_pos)
        
        # Add driver name and result with color coding
        position = int(strategy['Result'])  # Ensure integer
        position_color = '#ff4b4b' if position <= 3 else '#666666'
        y_labels.append(f"{strategy['Year']} - {strategy['Driver']} (P{position})")
        
        # Plot compounds with better styling
        for stint in strategy['Stints']:
            start = int(stint['start_lap'])  # Ensure integer
            end = int(stint['end_lap'])  # Ensure integer
            compound = stint['compound']
            
            # Plot stint bar with rounded corners
            ax1.barh(y_pos, end - start + 1, left=start, height=1.0,
                    color=compounds_colors.get(compound, 'gray'), alpha=0.9,
                    edgecolor='#333333', linewidth=0.5)
            
            # Add compound label with contrasting text color
            mid_point = start + (end - start) / 2
            text_color = 'black' if compound in ['MEDIUM', 'HARD'] else 'white'
            
            # Only add text if the stint is wide enough for text
            if end - start > 3:
                ax1.text(mid_point, y_pos, compound[0] if compound else '?',
                        ha='center', va='center', color=text_color, 
                        fontweight='bold', fontsize=12)
        
        # Plot pit stops with enhanced visibility
        for pit_lap in strategy['PitLaps']:
            pit_lap = int(pit_lap)  # Ensure integer
            # Add vertical line for pit stop
            ax1.axvline(x=pit_lap, ymin=(y_pos-0.6)/len(strategies_df)/6,
                       ymax=(y_pos+0.6)/len(strategies_df)/6,
                       color='black', linestyle='-', alpha=0.9, linewidth=1.5)
            
            # Add more visible marker at pit stop point
            ax1.plot([pit_lap], [y_pos], 'wo', markersize=8, alpha=1.0, 
                    markeredgecolor='black', markeredgewidth=1)
    
    # Customize strategy timeline with better styling
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(y_labels, fontsize=11, fontweight='medium')
    ax1.set_xlabel('Lap Number', fontsize=12, fontweight='bold')
    ax1.set_title(f'Historical Pit Stop Strategies - {team} at {race_name}',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, color='gray', linestyle=':')
    
    # Add x-axis gridlines at 10-lap intervals
    max_lap = max([int(stint['end_lap']) for strategy in strategies_df.itertuples() 
                  for stint in strategy.Stints]) if not strategies_df.empty else 60
    ax1.set_xticks(range(0, max_lap + 10, 10))
    
    # Move legend outside the plot for better visibility
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, label=compound, 
                                    edgecolor='black', linewidth=0.5)
                      for compound, color in compounds_colors.items()]
    ax1.legend(handles=legend_elements, loc='upper center', ncol=5,
              fontsize=10, bbox_to_anchor=(0.5, -0.15),
              facecolor='white', edgecolor='gray')
    
    # Plot lap time comparison with better styling
    bar_width = 0.8
    for i, (_, strategy) in enumerate(strategies_df.iterrows()):
        if 'FastestLap' in strategy and pd.notnull(strategy['FastestLap']):
            lap_time_seconds = strategy['FastestLap'].total_seconds()
            
            # Get the compound color for the fastest lap (usually the last compound)
            bar_color = compounds_colors.get(strategy['Compounds'][-1] 
                                           if strategy['Compounds'] else 'MEDIUM', 'gray')
            
            # Plot bar with better styling
            ax2.bar(i, lap_time_seconds, width=bar_width,
                   color=bar_color, alpha=0.85, 
                   edgecolor='black', linewidth=0.5)
            
            # Add time label with better font
            ax2.text(i, lap_time_seconds + 0.3, f"{lap_time_seconds:.1f}s",
                    ha='center', va='bottom', color='black', 
                    fontsize=10, fontweight='bold')
    
    # Customize lap time comparison with better styling
    ax2.set_xticks(range(len(strategies_df)))
    ax2.set_xticklabels([f"{s['Year']} - {s['Driver']}" for _, s in strategies_df.iterrows()],
                        rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Fastest Lap Time (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Fastest Lap Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, color='gray', linestyle=':')
    
    # Set y-axis limits with some padding
    if len(strategies_df) > 0:
        fastest_times = [s['FastestLap'].total_seconds() for _, s in strategies_df.iterrows() 
                        if 'FastestLap' in s and pd.notnull(s['FastestLap'])]
        if fastest_times:
            min_time = min(fastest_times)
            max_time = max(fastest_times)
            padding = (max_time - min_time) * 0.1 if max_time > min_time else 1.0
            ax2.set_ylim(min_time - padding, max_time + padding * 3)  # More padding on top for labels
    
    # Set x-axis limits for strategy timeline with some padding
    if len(strategies_df) > 0:
        max_lap = max([int(stint['end_lap']) for strategy in strategies_df.itertuples() 
                      for stint in strategy.Stints])
        ax1.set_xlim(0, max_lap + 5)  # Add some padding
    else:
        ax1.set_xlim(0, 60)  # Default if no data
    
    # Adjust layout for better spacing
    plt.tight_layout(pad=3.0)
    
    return fig

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
                
                # Create a mapping of known 2025 driver codes to teams
                driver_team_mapping = {
                    "PIA": "McLaren",
                    "NOR": "McLaren",
                    "VER": "Red Bull",
                    "RUS": "Mercedes",
                    "LEC": "Ferrari",
                    "HAM": "Ferrari",  # Lewis Hamilton moves to Ferrari [1, 2, 5, 6, 8, 9, 10]
                    "SAI": "Williams",  # Carlos Sainz moves to Williams [1, 5, 6, 10]
                    "PER": "Red Bull", # Sergio Pérez was dropped from Red Bull [1, 2, 5, 7] - Note: Some sources list Lawson or Tsunoda at Red Bull, but recent results confirm Tsunoda at Red Bull and Lawson at Racing Bulls [1, 2, 3, 6, 7, 9]
                    "ALB": "Williams",
                    "STR": "Aston Martin",
                    "TSU": "Red Bull",  # Yuki Tsunoda confirmed at Red Bull [1, 2, 3, 6, 9] - Note: Some sources initially listed Lawson at Red Bull, but this was updated [1, 2]
                    "HUL": "Kick Sauber",  # Nico Hulkenberg moves to Sauber (Kick Sauber) [1, 5, 6, 8, 9, 10]
                    "ALO": "Aston Martin",
                    "OCO": "Haas",  # Esteban Ocon moves to Haas [1, 3, 5, 6, 8, 9, 10]
                    "GAS": "Alpine",
                    "ZHO": "Reserve",  # Zhou Guanyu is listed as a reserve driver [1]
                    "RIC": "Racing Bulls", # Daniel Ricciardo is not listed in the confirmed 2025 lineups [2]
                    "BOT": "Reserve",  # Valtteri Bottas is listed as a reserve driver [1]
                    "LAW": "Racing Bulls",  # Liam Lawson is confirmed at Racing Bulls [2, 3, 6, 7, 9]
                    "ANT": "Mercedes",  # Kimi Antonelli joins Mercedes [1, 2, 3, 5, 6, 7, 8, 9, 10]
                    "BEA": "Haas",  # Oliver Bearman joins Haas [2, 3, 5, 6, 8, 9, 10]
                    "DOO": "Alpine",  # Jack Doohan joins Alpine [2, 3, 5, 6, 8, 9]
                    "BOR": "Kick Sauber",  # Gabriel Bortoleto joins Sauber (Kick Sauber) [2, 3, 5, 6, 8, 9]
                    "HAD": "Racing Bulls"  # Isack Hadjar joins Racing Bulls [2, 3, 5, 6, 7, 9]
                }

                
                for entry in standings_data["standings"].get("entries", []):
                    if "athlete" in entry:
                        driver = entry.get("athlete", {})
                        driver_code = driver.get("abbreviation", "")
                        
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
                            elif "Racing Bulls" in entry_str:
                                team_name = "Racing Bulls"
                        
                        # If team is still not found, use our mapping
                        if not team_name and driver_code in driver_team_mapping:
                            team_name = driver_team_mapping[driver_code]
                        
                        logger.info(f"Driver: {driver_code}, Team found: {team_name}")
                        
                        standings.append({
                            "Position": entry.get("stats", [])[0].get("displayValue", "-") if entry.get("stats") else "-",
                            "Driver": driver_code,
                            "Team": team_name,
                            "Points": entry.get("stats", [])[1].get("displayValue", "0") if len(entry.get("stats", [])) > 1 else "0"
                        })
                
                # Sort by position
                sorted_standings = sorted(standings, key=lambda x: int(x["Position"]) if x["Position"].isdigit() else 999)
                return pd.DataFrame(sorted_standings)
        
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
        
        # Placeholder data with 2025 drivers and teams
        return pd.DataFrame([
            {'Position': 1, 'Driver': 'PIA', 'Team': 'McLaren', 'Points': 131},
            {'Position': 2, 'Driver': 'NOR', 'Team': 'McLaren', 'Points': 115},
            {'Position': 3, 'Driver': 'VER', 'Team': 'Red Bull', 'Points': 99},
            {'Position': 4, 'Driver': 'RUS', 'Team': 'Mercedes', 'Points': 93},
            {'Position': 5, 'Driver': 'LEC', 'Team': 'Ferrari', 'Points': 53}
        ])
    except Exception as e:
        logger.error(f"Error getting driver standings: {str(e)}")
        # Return placeholder data with 2025 drivers and teams
        return pd.DataFrame([
            {'Position': 1, 'Driver': 'PIA', 'Team': 'McLaren', 'Points': 131},
            {'Position': 2, 'Driver': 'NOR', 'Team': 'McLaren', 'Points': 115},
            {'Position': 3, 'Driver': 'VER', 'Team': 'Red Bull', 'Points': 99},
            {'Position': 4, 'Driver': 'RUS', 'Team': 'Mercedes', 'Points': 93},
            {'Position': 5, 'Driver': 'LEC', 'Team': 'Ferrari', 'Points': 53}
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
            schedule_data = rapidapi_request("schedule", {"year": year})
            
            if schedule_data:
                events = []
                today = datetime.now().date()
                
                # Circuit to location mapping
                circuit_locations = {
                    "Melbourne Grand Prix Circuit": "Melbourne, Australia",
                    "Shanghai International Circuit": "Shanghai, China",
                    "Suzuka International Racing Course": "Suzuka, Japan",
                    "Bahrain International Circuit": "Sakhir, Bahrain",
                    "Jeddah Street Circuit": "Jeddah, Saudi Arabia",
                    "Miami International Autodrome": "Miami, USA",
                    "Autodromo Enzo e Dino Ferrari": "Imola, Italy",
                    "Circuit de Monaco": "Monte Carlo, Monaco",
                    "Circuit de Barcelona-Catalunya": "Barcelona, Spain",
                    "Circuit Gilles-Villeneuve": "Montreal, Canada",
                    "Red Bull Ring": "Spielberg, Austria",
                    "Silverstone Circuit": "Silverstone, UK",
                    "Hungaroring": "Budapest, Hungary",
                    "Circuit de Spa-Francorchamps": "Spa, Belgium",
                    "Circuit Park Zandvoort": "Zandvoort, Netherlands",
                    "Autodromo Nazionale Monza": "Monza, Italy",
                    "Baku City Circuit": "Baku, Azerbaijan",
                    "Marina Bay Street Circuit": "Singapore",
                    "Circuit of the Americas": "Austin, USA",
                    "Autodromo Hermanos Rodriguez": "Mexico City, Mexico",
                    "Autodromo Jose Carlos Pace": "São Paulo, Brazil",
                    "Las Vegas Street Circuit": "Las Vegas, USA",
                    "Losail International Circuit": "Doha, Qatar",
                    "Yas Marina Circuit": "Abu Dhabi, UAE"
                }
                
                # The API returns dates as keys (e.g., "20250313")
                for date_key in schedule_data:
                    # Each date can have multiple events, typically just one race
                    for event in schedule_data[date_key]:
                        # Extract event data
                        gp_name = event.get("gPrx", "")
                        circuit = event.get("crct", "")
                        completed = event.get("completed", False)
                        
                        # Determine location from circuit name using mapping
                        location = circuit_locations.get(circuit, "")
                        
                        # If not in mapping, try to extract from circuit name
                        if not location and circuit:
                            # Some circuits have city names as first word
                            parts = circuit.split()
                            if parts:
                                # If first word is "Circuit", try second word
                                if parts[0].lower() == "circuit":
                                    location = parts[1] if len(parts) > 1 else ""
                                else:
                                    location = parts[0]
                        
                        # Parse date from startDate
                        start_date_str = event.get("startDate", "")
                        try:
                            event_date = datetime.strptime(start_date_str, "%Y-%m-%dT%H:%MZ").date()
                            formatted_date = event_date.strftime("%d %b %Y")
                        except (ValueError, TypeError):
                            # If date parsing fails, use the date from the key
                            try:
                                key_date = datetime.strptime(date_key, "%Y%m%d").date()
                                formatted_date = key_date.strftime("%d %b %Y")
                                event_date = key_date
                            except ValueError:
                                formatted_date = date_key
                                event_date = today  # Fallback
                        
                        # Determine winner if available
                        winner = event.get("winner", "")
                        
                        events.append({
                            "SortDate": event_date,  # Add this for sorting
                            "Name": gp_name,
                            "Circuit": circuit,
                            "Location": location,
                            "Date": formatted_date,
                            "Status": "Completed" if completed else "Upcoming",
                            "Winner": winner
                        })
                
                # Sort events by date
                events_sorted = sorted(events, key=lambda x: x["SortDate"])
                
                # Assign round numbers sequentially based on sorted dates
                for i, event in enumerate(events_sorted, 1):
                    event["Round"] = str(i)
                    # Remove the temporary sort date key
                    event.pop("SortDate")
                
                return pd.DataFrame(events_sorted)
        
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
                    try:
                        st.image(item['image_url'])
                    except Exception as e:
                        logger.warning(f"Failed to load news image: {str(e)}")
                        # Skip image display on error
                        pass
                
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

def simulate_pit_stop_strategy(race_name, driver_code, weather_conditions, track_conditions, tire_choices):
    """Simulate a pit stop strategy for a given race and driver."""
    try:
        # Get historical data for this race
        historical_data = get_historical_strategy(race_name, driver_code)
        
        # Get current season data
        current_standings = get_driver_standings()
        current_team = ""
        try:
            # Get the driver's team if available
            driver_row = current_standings[current_standings['Driver'] == driver_code]
            if not driver_row.empty:
                current_team = driver_row['Team'].iloc[0]
        except Exception as e:
            logger.warning(f"Could not get team for driver {driver_code}: {e}")
            current_team = "Unknown Team"
        
        # Create a race map to get lap count
        race_laps = {
            'Monaco Grand Prix': 78,
            'Singapore Grand Prix': 62,
            'Bahrain Grand Prix': 57,
            'Abu Dhabi Grand Prix': 58,
            'Australian Grand Prix': 58,
            'Emilia Romagna Grand Prix': 63,
            'Miami Grand Prix': 57,
            'Japanese Grand Prix': 53,
            'Chinese Grand Prix': 56,
            'United States Grand Prix': 56,
            'Spanish Grand Prix': 66,
            'Austrian Grand Prix': 71,
            'British Grand Prix': 52,
            'Hungarian Grand Prix': 70,
            'Belgian Grand Prix': 44,
            'Dutch Grand Prix': 72,
            'Italian Grand Prix': 53,
            'Azerbaijan Grand Prix': 51,
            'Qatar Grand Prix': 57,
            'Brazilian Grand Prix': 71,
            'Las Vegas Grand Prix': 50,
            'Mexican Grand Prix': 71,
            'Canadian Grand Prix': 70,
            'Saudi Arabian Grand Prix': 50
        }
        
        # Get total laps for this race or use default
        total_laps = race_laps.get(race_name, 60)
        
        # Base simulation on historical data and current conditions
        simulation_results = {
            'race_name': race_name,
            'driver': driver_code,
            'team': current_team,
            'weather': weather_conditions,
            'track': track_conditions,
            'tire_choices': tire_choices,
            'total_laps': total_laps,
            'predicted_stops': [],
            'stints': []
        }
        
        # Calculate base number of stops based on historical data
        avg_stops = 1  # Default value if no historical data
        if not historical_data.empty:
            avg_stops = historical_data['NumStops'].mean()
        
        # Adjust based on weather and track conditions
        if weather_conditions == 'Rain':
            avg_stops += 1
        elif weather_conditions == 'Mixed':
            avg_stops += 0.5
            
        if track_conditions == 'High Degradation':
            avg_stops += 0.5
        elif track_conditions == 'Low Degradation':
            avg_stops -= 0.5
        
        # Calculate stops (minimum 1, maximum based on tire choices)
        stops = min(max(round(avg_stops), 1), len(tire_choices))
        
        # Calculate base lap time (seconds)
        base_lap_time = 90
        degradation_factor = 0.1  # seconds per lap
        
        if track_conditions == 'High Degradation':
            degradation_factor = 0.2
        elif track_conditions == 'Low Degradation':
            degradation_factor = 0.05
            
        # Generate stop laps based on tire wear
        stint_lengths = []
        
        # Tire-specific max stint lengths
        max_stint_lengths = {
            'SOFT': int(total_laps * 0.3),
            'MEDIUM': int(total_laps * 0.5),
            'HARD': int(total_laps * 0.7),
            'INTERMEDIATE': int(total_laps * 0.4) if weather_conditions in ['Rain', 'Mixed'] else int(total_laps * 0.3),
            'WET': int(total_laps * 0.5) if weather_conditions == 'Rain' else int(total_laps * 0.25)
        }
        
        remaining_laps = total_laps
        current_lap = 0
        
        # Create stints based on tire choices
        for i in range(stops):
            if i >= len(tire_choices):
                break
                
            tire = tire_choices[i]
            
            # Calculate max possible stint length based on tire compound
            max_length = max_stint_lengths.get(tire, int(total_laps * 0.4))
            
            # If it's the last stint, use all remaining laps
            if i == stops - 1:
                stint_length = remaining_laps
            else:
                # Calculate a realistic stint length, between 15 laps and max for that compound
                min_stint = min(15, remaining_laps // 2)
                max_stint = min(max_length, remaining_laps - 10)  # Keep at least 10 laps for the last stint
                
                # Randomize a bit to make it more realistic
                import random
                stint_length = random.randint(min_stint, max_stint)
            
            stint_lengths.append(stint_length)
            remaining_laps -= stint_length
            
            # Add pit stop data (except for the last one)
            if i < stops - 1:
                stop_lap = current_lap + stint_length
                
                simulation_results['predicted_stops'].append({
                    'lap': stop_lap,
                    'tire_from': tire,
                    'tire_to': tire_choices[i+1] if i+1 < len(tire_choices) else 'UNKNOWN',
                    'estimated_time': base_lap_time + (stop_lap * degradation_factor)
                })
            
            # Add stint data
            simulation_results['stints'].append({
                'start_lap': current_lap,
                'end_lap': current_lap + stint_length,
                'compound': tire,
                'avg_lap_time': base_lap_time + (current_lap + stint_length/2) * degradation_factor
            })
            
            current_lap += stint_length
        
        return simulation_results
    except Exception as e:
        logger.error(f"Error in pit stop simulation: {str(e)}")
        return None

def create_simulation_interface():
    """Create the simulation interface."""
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">🏎️ Pit Stop Strategy Simulator</h2>', unsafe_allow_html=True)
    
    # Get available races
    schedule = get_full_season_schedule(2025)
    upcoming_races = schedule[schedule['Status'] == 'Upcoming']['Name'].tolist()
    
    # Fallback if no upcoming races found
    if not upcoming_races:
        upcoming_races = [
            "Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix",
            "Japanese Grand Prix", "Chinese Grand Prix", "Miami Grand Prix",
            "Emilia Romagna Grand Prix", "Monaco Grand Prix", "Canadian Grand Prix",
            "Spanish Grand Prix", "Austrian Grand Prix", "British Grand Prix",
            "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix",
            "Italian Grand Prix", "Azerbaijan Grand Prix", "Singapore Grand Prix",
            "United States Grand Prix", "Mexican Grand Prix", "Brazilian Grand Prix",
            "Las Vegas Grand Prix", "Qatar Grand Prix", "Abu Dhabi Grand Prix"
        ]
    
    # Create input columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Race selection
        selected_race = st.selectbox(
            "Select Race",
            upcoming_races
        )
        
        # Driver selection
        driver_standings = get_driver_standings()
        drivers = driver_standings['Driver'].tolist()
        selected_driver = st.selectbox(
            "Select Driver",
            drivers
        )
        
        # Weather conditions
        weather_options = ['Dry', 'Rain', 'Mixed']
        selected_weather = st.selectbox(
            "Weather Conditions",
            weather_options
        )
    
    with col2:
        # Track conditions
        track_options = ['High Degradation', 'Medium Degradation', 'Low Degradation']
        selected_track = st.selectbox(
            "Track Conditions",
            track_options
        )
        
        # Tire choices
        st.write("Select Tire Strategy (in order of use):")
        tire_choices = []
        for i in range(3):  # Allow up to 3 different tire compounds
            tire = st.selectbox(
                f"Tire {i+1}",
                ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'],
                key=f"tire_{i}"
            )
            if tire:
                tire_choices.append(tire)
    
    # Run simulation button
    if st.button("Run Simulation", key="run_simulation", use_container_width=True, type="primary"):
        with st.spinner("Simulating strategy..."):
            results = simulate_pit_stop_strategy(
                selected_race,
                selected_driver,
                selected_weather,
                selected_track,
                tire_choices
            )
            
            if results:
                # Display results
                st.subheader("Simulation Results")
                
                # Create columns for results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Race Information")
                    st.write(f"**Race:** {results['race_name']}")
                    st.write(f"**Driver:** {results['driver']} ({results['team']})")
                    st.write(f"**Weather:** {results['weather']}")
                    st.write(f"**Track:** {results['track']}")
                    st.write(f"**Total Laps:** {results['total_laps']}")
                
                with col2:
                    st.markdown("### Predicted Strategy")
                    if results['predicted_stops']:
                        for i, stop in enumerate(results['predicted_stops']):
                            st.write(f"**Pit Stop {i+1} (Lap {stop['lap']}):** {stop['tire_from']} → {stop['tire_to']}")
                            st.write(f"Estimated lap time: {stop['estimated_time']:.2f}s")
                    else:
                        st.write("**No pit stops predicted - one-stop strategy**")
                    
                    # Show stint summary
                    st.markdown("### Stint Summary")
                    for i, stint in enumerate(results['stints']):
                        st.write(f"**Stint {i+1}:** Laps {stint['start_lap']+1}-{stint['end_lap']} on {stint['compound']}")
                        st.write(f"Length: {stint['end_lap'] - stint['start_lap']} laps")
                
                # Add visualization
                st.markdown("### Strategy Timeline")
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Plot tire stints
                tire_colors = {
                    'SOFT': '#FF1E1E',
                    'MEDIUM': '#FFF200',
                    'HARD': '#FFFFFF',
                    'INTERMEDIATE': '#39B54A',
                    'WET': '#00A0DC'
                }
                
                # Plot each stint as a horizontal bar
                for stint in results['stints']:
                    ax.barh(0.5, stint['end_lap'] - stint['start_lap'], left=stint['start_lap'], 
                           color=tire_colors.get(stint['compound'], 'gray'),
                           alpha=0.8, height=0.6)
                    
                    # Add compound label in the middle of the stint
                    mid_lap = stint['start_lap'] + (stint['end_lap'] - stint['start_lap']) / 2
                    text_color = 'black' if stint['compound'] in ['MEDIUM', 'HARD'] else 'white'
                    ax.text(mid_lap, 0.5, stint['compound'],
                           ha='center', va='center', fontweight='bold',
                           color=text_color)
                
                # Add pit stop markers
                for stop in results['predicted_stops']:
                    ax.axvline(x=stop['lap'], color='black', linestyle='--', linewidth=2)
                    ax.plot(stop['lap'], 0.5, 'ko', markersize=10)
                
                # Add lap counter on bottom
                lap_markers = list(range(0, results['total_laps'] + 1, 10))
                if results['total_laps'] not in lap_markers:
                    lap_markers.append(results['total_laps'])
                    
                ax.set_xticks(lap_markers)
                ax.set_xlabel('Lap Number', fontsize=12)
                
                # Hide y-axis labels since we only have one row
                ax.set_yticks([])
                
                # Set x-axis limits
                ax.set_xlim(-1, results['total_laps'] + 1)
                
                # Add title and grid
                ax.set_title(f'Pit Stop Strategy - {results["driver"]} at {results["race_name"]}', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                # Add legend for tire compounds
                legend_elements = [plt.Rectangle((0,0), 1, 1, fc=color, label=compound) 
                                 for compound, color in tire_colors.items()]
                ax.legend(handles=legend_elements, loc='upper center', 
                         bbox_to_anchor=(0.5, -0.15), ncol=5)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add race simulation visualization
                st.markdown("### Lap Time Simulation")
                
                # Create lap time simulation
                lap_times = []
                positions = []
                current_position = random.randint(5, 15)  # Start position
                
                for lap in range(1, results['total_laps'] + 1):
                    # Find which stint this lap belongs to
                    current_stint = None
                    for stint in results['stints']:
                        if stint['start_lap'] < lap <= stint['end_lap']:
                            current_stint = stint
                            break
                    
                    if current_stint:
                        # Calculate lap time with some random variation
                        base_time = current_stint['avg_lap_time']
                        variation = random.uniform(-0.5, 0.5)
                        
                        # Add pit stop delay
                        pit_stop_delay = 0
                        for stop in results['predicted_stops']:
                            if stop['lap'] == lap:
                                pit_stop_delay = random.uniform(20, 24)  # Pit stop takes ~22 seconds
                        
                        lap_time = base_time + variation + pit_stop_delay
                        lap_times.append(lap_time)
                        
                        # Update position (improve it slightly over the race with some randomness)
                        position_change = 0
                        if pit_stop_delay > 0:
                            # Lose positions in pit
                            position_change = random.randint(1, 3)
                        else:
                            # Randomly gain or lose positions
                            position_change = random.choice([-1, -1, 0, 0, 0, 1])
                        
                        current_position = max(1, min(20, current_position + position_change))
                        positions.append(current_position)
                
                # Create a figure with two subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
                
                # Plot lap times
                ax1.plot(range(1, results['total_laps'] + 1), lap_times, 'b-', alpha=0.7)
                
                # Add points for pit stops
                for stop in results['predicted_stops']:
                    stop_index = stop['lap'] - 1
                    if 0 <= stop_index < len(lap_times):
                        ax1.plot(stop['lap'], lap_times[stop_index], 'ro', markersize=8)
                        ax1.text(stop['lap'], lap_times[stop_index] + 1, f"Pit\n{stop['tire_from']} → {stop['tire_to']}", 
                                ha='center', va='bottom', fontsize=8)
                
                # Customize lap time plot
                ax1.set_xlabel('Lap')
                ax1.set_ylabel('Lap Time (seconds)')
                ax1.set_title('Simulated Lap Times')
                ax1.grid(True, alpha=0.3)
                
                # Add stint background colors
                for stint in results['stints']:
                    ax1.axvspan(stint['start_lap'], stint['end_lap'], 
                               alpha=0.1, 
                               color=tire_colors.get(stint['compound'], 'gray'))
                
                # Plot position
                ax2.plot(range(1, results['total_laps'] + 1), positions, 'g-', alpha=0.7)
                ax2.set_ylim(20.5, 0.5)  # Reverse y-axis to show position 1 at the top
                ax2.set_xlabel('Lap')
                ax2.set_ylabel('Position')
                ax2.set_title('Simulated Race Position')
                ax2.grid(True, alpha=0.3)
                
                # Mark pit stops on position chart
                for stop in results['predicted_stops']:
                    stop_index = stop['lap'] - 1
                    if 0 <= stop_index < len(positions):
                        ax2.plot(stop['lap'], positions[stop_index], 'ro', markersize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Add race summary
                final_position = positions[-1] if positions else current_position
                st.markdown("### Race Summary")
                st.write(f"**Predicted Finish Position:** P{final_position}")
                st.write(f"**Total Pit Stops:** {len(results['predicted_stops'])}")
                
                # Calculate total race time
                total_time = sum(lap_times)
                hours = int(total_time // 3600)
                minutes = int((total_time % 3600) // 60)
                seconds = total_time % 60
                st.write(f"**Estimated Race Time:** {hours}h {minutes}m {seconds:.3f}s")
            else:
                st.error("Failed to generate simulation. Please try again.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="PitGenius - F1 Pit Stop Predictions", layout="wide")
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    /* Enhance the plots and tables */
    .predictions-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    
    /* Better styling for strategy summary */
    .strategy-summary {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Style the strategy table */
    .strategy-table {
        width: 100%;
        margin-top: 15px;
    }
    
    /* Style section headers */
    .section-header {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #ff1e1e;
        color: #333333;
    }
    
    /* Style selectors and inputs */
    .stSelectbox {
        margin-bottom: 15px;
    }
    
    /* General page styling */
    .main {
        padding: 2rem;
    }
    
    /* Larger plot container */
    .element-container.st-emotion-cache-1r6slb0.e1f1d6gn2 {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description in the header
    st.title("🏎️ PitGenius: F1 Pit Stop Predictions")
    
    # Initialize session state for navigation
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = "📊 Dashboard"
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Navigation Options",
        ["📊 Dashboard", "🔮 Predictions", "📈 Historical Analysis", "🎮 Strategy Simulator"],
        index=["📊 Dashboard", "🔮 Predictions", "📈 Historical Analysis", "🎮 Strategy Simulator"].index(st.session_state.app_mode),
        label_visibility="collapsed"
    )
    
    # Update session state
    st.session_state.app_mode = app_mode
    
    # Sidebar info/branding
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "PitGenius uses machine learning to predict F1 pit stop strategies "
        "based on historical data from the 2020-2023 seasons."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Developed with")
    st.sidebar.markdown("- FastF1, RapidAPI, and Ergast API 📊")
    st.sidebar.markdown("- Scikit-learn 🤖")
    st.sidebar.markdown("- Streamlit 📊")
    
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
    
    # Show appropriate content based on the selected mode
    if app_mode == "📊 Dashboard":
        create_dashboard()
    elif app_mode == "🎮 Strategy Simulator":
        create_simulation_interface()
    else:
        # Get available races for predictions and historical analysis
        schedule = fastf1.get_event_schedule(2025)
        races = schedule[schedule['EventFormat'] == 'conventional']['EventName'].tolist()
        
        # For predictions and historical analysis modes, show selection controls at top
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Race selection
            # Find British GP in the list of races, if not found default to first race
            default_race_index = 0
            for idx, race in enumerate(races):
                if "British" in race:
                    default_race_index = idx
                    break
            
            selected_race = st.selectbox(
                "Select Race",
                races,
                index=default_race_index
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
                            # Add a container with max width and better styling
                            with st.container():
                                st.markdown('<div class="predictions-container">', unsafe_allow_html=True)
                                st.markdown('<h3 class="section-header">Pit Stop Prediction Chart</h3>', unsafe_allow_html=True)
                                
                                # Create visualization
                                fig = plot_driver_prediction(predictions, selected_driver)
                                st.pyplot(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            # Display strategy summary with better styling
                            st.markdown('<div class="strategy-summary">', unsafe_allow_html=True)
                            st.markdown('<h3 class="section-header">Predicted Strategy Summary</h3>', unsafe_allow_html=True)
                            
                            driver_preds = predictions[
                                (predictions['Driver'] == selected_driver) & 
                                predictions['PredictedPitStop']
                            ]
                            
                            if not driver_preds.empty:
                                st.markdown(f"<p><strong>Number of predicted pit stops:</strong> {len(driver_preds)}</p>", unsafe_allow_html=True)
                                
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
                                    st.markdown('<div class="strategy-table">', unsafe_allow_html=True)
                                    st.table(pd.DataFrame(strategy_data))
                                    st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.warning("No pit stops predicted for this driver.")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add driver details card
                            st.markdown('<div class="strategy-summary" style="margin-top: 20px;">', unsafe_allow_html=True)
                            st.markdown('<h3 class="section-header">Driver Details</h3>', unsafe_allow_html=True)
                            
                            # Find driver's full name
                            driver_name = next((d['name'] for d in team_drivers if d['code'] == selected_driver), selected_driver)
                            
                            st.markdown(f"<p><strong>Driver:</strong> {driver_name}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p><strong>Team:</strong> {selected_team}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p><strong>Race:</strong> {selected_race}</p>", unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
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
                        # Create two columns with improved width ratio
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Add a container with better styling
                            with st.container():
                                st.markdown('<div class="predictions-container">', unsafe_allow_html=True)
                                st.markdown('<h3 class="section-header">Historical Pit Stop Strategies</h3>', unsafe_allow_html=True)
                                
                                # Plot historical strategies
                                fig = plot_historical_strategies(historical_data, selected_race, selected_team)
                                st.pyplot(fig, use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            # Display summary statistics with better styling
                            st.markdown('<div class="strategy-summary">', unsafe_allow_html=True)
                            st.markdown('<h3 class="section-header">Historical Strategy Summary</h3>', unsafe_allow_html=True)
                            
                            avg_stops = historical_data['NumStops'].mean()
                            st.markdown(f"<p><strong>Average number of pit stops:</strong> {avg_stops:.1f}</p>", unsafe_allow_html=True)
                            
                            # Most common compounds with better styling
                            all_compounds = [compound for compounds in historical_data['Compounds'] for compound in compounds]
                            if all_compounds:
                                compound_counts = pd.Series(all_compounds).value_counts()
                                st.markdown("<p><strong>Most used tire compounds:</strong></p>", unsafe_allow_html=True)
                                
                                # Display compounds using Streamlit components instead of raw HTML
                                for compound, count in compound_counts.items():
                                    color = "#FF1E1E" if compound == "SOFT" else \
                                            "#FFF200" if compound == "MEDIUM" else \
                                            "#FFFFFF" if compound == "HARD" else \
                                            "#39B54A" if compound == "INTERMEDIATE" else \
                                            "#00A0DC" if compound == "WET" else "gray"
                                    
                                    text_color = "white" if compound in ["SOFT", "WET", "INTERMEDIATE"] else "black"
                                    
                                    st.markdown(
                                        f"""<div style="display:flex; align-items:center; margin-bottom:8px;">
                                            <span style="display:inline-block; width:100px; background-color:{color}; 
                                            color:{text_color}; padding:3px 6px; border-radius:4px; 
                                            text-align:center; margin-right:10px; font-weight:bold;">{compound}</span>
                                            <span style="font-weight:bold;">{count} times</span>
                                        </div>""", 
                                        unsafe_allow_html=True
                                    )
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Race Details card
                            st.markdown('<div class="strategy-summary" style="margin-top: 20px;">', unsafe_allow_html=True)
                            st.markdown('<h3 class="section-header">Race Details</h3>', unsafe_allow_html=True)
                            
                            st.markdown(f"<p><strong>Team:</strong> {selected_team}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p><strong>Race:</strong> {selected_race}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p><strong>Seasons Analyzed:</strong> 2022-2024</p>", unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display detailed data in a full-width container below
                        st.markdown('<div class="predictions-container" style="margin-top: 20px;">', unsafe_allow_html=True)
                        st.markdown('<h3 class="section-header">Detailed Historical Data</h3>', unsafe_allow_html=True)
                        
                        # Create a styled dataframe
                        display_data = historical_data[['Year', 'Driver', 'NumStops', 'Result']].copy()
                        display_data = display_data.sort_values(['Year', 'Result'])
                        
                        # Rename columns for better display
                        display_data.columns = ['Season', 'Driver', 'Pit Stops', 'Result']
                        
                        # Show the dataframe with custom formatting
                        st.dataframe(
                            display_data,
                            column_config={
                                "Season": st.column_config.NumberColumn("Season", format="%d"),
                                "Driver": st.column_config.TextColumn("Driver", width="medium"),
                                "Pit Stops": st.column_config.NumberColumn("Pit Stops", format="%d"),
                                "Result": st.column_config.NumberColumn("Final Position", format="P%d")
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"No historical data found for {selected_team} at {selected_race}")
                    
            except Exception as e:
                st.error(f"Error loading historical data: {str(e)}")
                logger.error(f"Error loading historical data: {str(e)}")
                
                # Provide more helpful error message with potential solutions
                st.markdown("""
                <div style="padding: 15px; background-color: #f8f8f8; border-left: 5px solid #ff9800; margin-top: 20px;">
                    <h4 style="color: #ff5722; margin-top: 0;">Troubleshooting Tips</h4>
                    <ul>
                        <li>Try selecting a different race or team combination</li>
                        <li>Check that FastF1 can access the selected race data</li>
                        <li>Some historical data may not be available for certain tracks or seasons</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 