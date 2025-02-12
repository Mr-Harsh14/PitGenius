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

# Add project root to path
if os.environ.get('STREAMLIT_SHARING'):
    project_root = '/mount/src/pitgenius'
else:
    project_root = str(Path(__file__).parent.parent)

if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.train_random_forest import load_model
from src.models.predict_pit_stops import get_race_data, prepare_features, predict_race_pit_stops

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastF1 cache
cache_dir = Path(project_root) / 'data' / 'raw' / 'fastf1_cache'
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

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
    
    for idx, lap in driver_predictions.iterrows():
        if lap['CurrentCompound'] != current_compound:
            if current_compound is not None:
                ax2.axvspan(start_lap, lap['LapNumber'], 
                           ymin=0.0, ymax=1.0,
                           color=compounds.get(current_compound, 'gray'), alpha=0.8)
                # Add compound label
                mid_lap = (start_lap + lap['LapNumber']) / 2
                text_color = 'black' if current_compound in ['MEDIUM', 'HARD'] else 'white'
                ax2.text(mid_lap, 0.5, current_compound[0], 
                        horizontalalignment='center', verticalalignment='center',
                        color=text_color, fontweight='bold')
            current_compound = lap['CurrentCompound']
            start_lap = lap['LapNumber']
    
    # Plot the last compound
    if current_compound is not None:
        ax2.axvspan(start_lap, max(driver_predictions['LapNumber']), 
                   ymin=0.0, ymax=1.0,
                   color=compounds.get(current_compound, 'gray'), alpha=0.8)
        mid_lap = (start_lap + max(driver_predictions['LapNumber'])) / 2
        text_color = 'black' if current_compound in ['MEDIUM', 'HARD'] else 'white'
        ax2.text(mid_lap, 0.5, current_compound[0], 
                horizontalalignment='center', verticalalignment='center',
                color=text_color, fontweight='bold')
    
    # Plot predicted pit stops on compound subplot
    for pit_lap in predicted_stops:
        ax2.axvline(x=pit_lap, color='white', linestyle='--', alpha=0.8, linewidth=2)
    
    # Customize compound subplot
    ax2.set_xlabel('Lap Number')
    ax2.set_ylabel('Predicted Compound')
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

def main():
    st.set_page_config(page_title="PitGenius - F1 Pit Stop Predictions", layout="wide")
    
    # Title and description
    st.title("🏎️ PitGenius: F1 Pit Stop Predictions")
    st.markdown("""
    Predict pit stop strategies for Formula 1 races using machine learning.
    Select a race and driver to see predicted pit stops and tire compounds.
    """)
    
    # Sidebar for inputs
    st.sidebar.header("Race Selection")
    
    # Get available races for 2024
    schedule = fastf1.get_event_schedule(2024)
    races = schedule[schedule['EventFormat'] == 'conventional']['EventName'].tolist()
    
    # Race selection
    selected_race = st.sidebar.selectbox(
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
    
    # Driver selection with team grouping
    st.sidebar.header("Driver Selection")
    
    # Group drivers by team
    teams = {}
    for driver in drivers_2024:
        if driver['team'] not in teams:
            teams[driver['team']] = []
        teams[driver['team']].append(driver)
    
    # Create team selection first
    selected_team = st.sidebar.selectbox(
        "Select Team",
        options=list(teams.keys())
    )
    
    # Then filter drivers by selected team
    team_drivers = teams[selected_team]
    selected_driver = st.sidebar.selectbox(
        "Select Driver",
        options=[d['code'] for d in team_drivers],
        format_func=lambda x: next(d['name'] for d in team_drivers if d['code'] == x)
    )
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔮 Predictions", "📊 Historical Analysis"])
    
    with tab1:
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
                    # Create two columns with different widths
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
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
                                next_compound = predictions[
                                    (predictions['Driver'] == selected_driver) & 
                                    (predictions['LapNumber'] > stop['LapNumber'])
                                ].iloc[0]['CurrentCompound']
                                
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
    
    with tab2:
        try:
            with st.spinner("Loading historical data..."):
                # Get historical strategies
                historical_data = get_historical_strategy(selected_race, selected_team)
                
                if not historical_data.empty:
                    # Create two columns
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
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