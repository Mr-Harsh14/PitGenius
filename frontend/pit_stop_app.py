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
import subprocess

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define 2024 F1 drivers
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

def setup_git_lfs():
    """Initialize Git LFS and pull the model file."""
    try:
        # Initialize Git LFS
        subprocess.run(['git', 'lfs', 'install'], check=True)
        # Pull LFS files
        subprocess.run(['git', 'lfs', 'pull'], check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting up Git LFS: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error setting up Git LFS: {str(e)}")
        return False

# Initialize Git LFS
if not setup_git_lfs():
    st.error("Failed to initialize Git LFS. Some features may not work correctly.")

# Now import the model-related modules
try:
    from src.models.train_random_forest import load_model
    from src.models.predict_pit_stops import get_race_data, prepare_features, predict_race_pit_stops
except ImportError as e:
    st.error(f"Error importing required modules: {str(e)}")
    st.info("Please make sure all dependencies are installed correctly.")
    st.stop()

# Initialize FastF1 cache
cache_dir = Path(project_root) / 'data' / 'raw' / 'fastf1_cache'
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

# App title and description
st.title('PitGenius: F1 Pit Stop Predictions')
st.write('Predict pit stop strategies for Formula 1 races using machine learning. Select a race and driver to see predicted pit stops and tire compounds.')

def check_model_exists():
    """Check if the model file exists and return appropriate message."""
    model_dir = Path(project_root) / 'models' / 'random_forest'
    model_file = model_dir / 'rf_model_2022-2023.joblib'
    
    if not model_file.exists():
        st.error("Model file not found. The model needs to be trained first.")
        st.info("""
        To train the model:
        1. Clone the repository locally
        2. Install dependencies: `pip install -r requirements.txt`
        3. Run the training script: `python src/models/train_random_forest.py`
        4. Push the trained model to the repository
        """)
        return False
        
    # Check file size to ensure it's not corrupted
    if model_file.stat().st_size < 1000000:  # Model should be at least 1MB
        st.error("Model file appears to be corrupted. Please retrain the model.")
        return False
        
    return True

@st.cache_resource
def load_trained_model():
    """Load the trained model (cached)"""
    if not check_model_exists():
        return None
        
    try:
        seasons = [2022, 2023]
        model = load_model(seasons)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        logger.error(f"Error loading model: {str(e)}")
        return None

# Load model
model = load_trained_model()

if model is None:
    st.warning("Please train the model before using the prediction interface.")
    st.stop()

# Race selection
st.header('Race Selection')
race = st.selectbox(
    'Select Race',
    ['Bahrain Grand Prix', 'Saudi Arabian Grand Prix', 'Australian Grand Prix',
     'Azerbaijan Grand Prix', 'Miami Grand Prix', 'Monaco Grand Prix',
     'Spanish Grand Prix', 'Canadian Grand Prix', 'Austrian Grand Prix',
     'British Grand Prix', 'Hungarian Grand Prix', 'Belgian Grand Prix',
     'Dutch Grand Prix', 'Italian Grand Prix', 'Singapore Grand Prix',
     'Japanese Grand Prix', 'Qatar Grand Prix', 'United States Grand Prix',
     'Mexico City Grand Prix', 'São Paulo Grand Prix', 'Las Vegas Grand Prix',
     'Abu Dhabi Grand Prix']
)

# Team selection
st.header('Driver Selection')
teams = sorted(list(set(driver['team'] for driver in drivers_2024)))
selected_team = st.selectbox('Select Team', teams)

# Filter drivers by selected team
team_drivers = [driver for driver in drivers_2024 if driver['team'] == selected_team]
selected_driver = st.selectbox(
    'Select Driver',
    [f"{driver['name']} ({driver['code']})" for driver in team_drivers]
)

if st.button('Analyze Race Strategy'):
    try:
        # Get race data
        race_data = get_race_data(2024, race)
        
        # Prepare features
        features = prepare_features(race_data)
        
        # Make predictions
        predictions = predict_race_pit_stops(model, features)
        
        # Display results
        st.header('Pit Stop Predictions')
        st.write(f"Analyzing {selected_driver}'s strategy for {race}")
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(predictions['lap'], predictions['pit_probability'])
        ax.set_title(f'Pit Stop Probability - {selected_driver} at {race}')
        ax.set_xlabel('Lap')
        ax.set_ylabel('Probability')
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Show detailed predictions
        st.subheader('Detailed Analysis')
        st.dataframe(predictions)
        
    except Exception as e:
        st.error(f"Error analyzing race strategy: {str(e)}")
        logger.error(f"Error analyzing race strategy: {str(e)}")

def plot_driver_prediction(predictions: pd.DataFrame, driver_code: str):
    """Create a visualization of predicted pit stops for a driver."""
    # Filter predictions for this driver
    driver_predictions = predictions[
        predictions['Driver'] == driver_code
    ].sort_values('LapNumber')
    
    if driver_predictions.empty:
        st.warning(f"No predictions found for driver {driver_code}")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1], gridspec_kw={'hspace': 0.1})
    
    # Plot pit stop probability on top subplot
    ax1.plot(driver_predictions['LapNumber'], driver_predictions['PitProbability'], 
             label='Pit Stop Probability', color='blue', alpha=0.8, linewidth=2)
    
    # Plot prediction threshold
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, 
                label='Prediction Threshold')
    
    # Plot predicted pit stops
    predicted_stops = driver_predictions[driver_predictions['PredictedPitStop']]['LapNumber']
    for pred_lap in predicted_stops:
        ax1.axvline(x=pred_lap, color='green', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add legend to top subplot
    ax1.legend(loc='upper right')
    
    # Customize top subplot
    ax1.set_title(f'Pit Stop Predictions - {driver_code}', pad=20)
    ax1.set_ylabel('Pit Stop Probability')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(driver_predictions['LapNumber']) + 1)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot predicted compounds on bottom subplot
    compounds = {
        'SOFT': 'salmon',
        'MEDIUM': 'yellow',
        'HARD': 'white',
        'INTERMEDIATE': 'lightgreen',
        'WET': 'lightblue'
    }
    
    # Plot predicted compounds
    current_compound = None
    start_lap = 0
    
    for idx, lap in driver_predictions.iterrows():
        if lap['CurrentCompound'] != current_compound:
            if current_compound is not None:
                ax2.axvspan(start_lap, lap['LapNumber'], 
                           ymin=0.0, ymax=1.0,
                           color=compounds.get(current_compound, 'gray'), alpha=0.3)
                # Add compound label
                mid_lap = (start_lap + lap['LapNumber']) / 2
                ax2.text(mid_lap, 0.5, current_compound[0], 
                        horizontalalignment='center', verticalalignment='center')
            current_compound = lap['CurrentCompound']
            start_lap = lap['LapNumber']
    
    # Plot the last compound
    if current_compound is not None:
        ax2.axvspan(start_lap, max(driver_predictions['LapNumber']), 
                   ymin=0.0, ymax=1.0,
                   color=compounds.get(current_compound, 'gray'), alpha=0.3)
        mid_lap = (start_lap + max(driver_predictions['LapNumber'])) / 2
        ax2.text(mid_lap, 0.5, current_compound[0], 
                horizontalalignment='center', verticalalignment='center')
    
    # Plot predicted pit stops on compound subplot
    for pit_lap in predicted_stops:
        ax2.axvline(x=pit_lap, color='green', linestyle='--', alpha=0.5)
    
    # Customize compound subplot
    ax2.set_xlabel('Lap Number')
    ax2.set_ylabel('Predicted Compound')
    ax2.set_yticks([])
    ax2.set_xlim(0, max(driver_predictions['LapNumber']) + 1)
    
    # Add compound legend
    compound_patches = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3, label=compound)
                       for compound, color in compounds.items()]
    ax2.legend(handles=compound_patches, loc='upper right', ncol=5)
    
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
                # Create visualization
                fig = plot_driver_prediction(predictions, selected_driver)
                
                # Display plot
                st.pyplot(fig)
                
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

if __name__ == "__main__":
    main() 