import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="PitGenius - F1 Pit Stop Predictions",
    layout="wide"
)

import logging
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import fastf1
import numpy as np
import subprocess
from datetime import datetime
import os

# Add project root to path
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

# Define 2024 F1 drivers and teams
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

def plot_pit_stop_predictions(predictions):
    """Create a visualization of pit stop predictions."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot pit stop probabilities
    ax.plot(predictions['lap'], predictions['pit_probability'], 
            color='blue', linewidth=2, label='Pit Stop Probability')
    
    # Add threshold line
    ax.axhline(y=0.4, color='red', linestyle='--', label='Decision Threshold (0.4)')
    
    # Highlight predicted pit stops
    pit_stops = predictions[predictions['predicted_pit']]
    if not pit_stops.empty:
        ax.scatter(pit_stops['lap'], pit_stops['pit_probability'], 
                  color='red', s=100, label='Predicted Pit Stops')
    
    # Customize plot
    ax.set_xlabel('Lap Number')
    ax.set_ylabel('Pit Stop Probability')
    ax.set_title('Pit Stop Predictions')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

def display_strategy_insights(predictions, driver_name):
    """Display pit stop strategy insights."""
    st.subheader('Strategy Insights')
    
    # Get predicted pit stops
    pit_stops = predictions[predictions['predicted_pit']]
    n_stops = len(pit_stops)
    
    if n_stops == 0:
        st.write("No pit stops predicted - likely a one-stop strategy.")
    else:
        st.write(f"Predicted {n_stops} pit stop{'s' if n_stops > 1 else ''} for {driver_name}")
        
        # Create a table of pit stop predictions
        strategy_df = pd.DataFrame({
            'Lap': pit_stops['lap'],
            'Probability': pit_stops['pit_probability'].map('{:.1%}'.format)
        }).reset_index(drop=True)
        
        strategy_df.index = [f"Stop {i+1}" for i in range(len(strategy_df))]
        st.table(strategy_df)
        
        # Add some strategy insights
        if n_stops == 1:
            st.write("💡 Classic one-stop strategy predicted.")
        elif n_stops == 2:
            st.write("💡 Two-stop strategy predicted - watch for undercut opportunities.")
        else:
            st.write("💡 Aggressive multi-stop strategy predicted - high tire degradation expected.")

def main():
    # Title and description
    st.title("🏎️ PitGenius: F1 Pit Stop Predictions")
    st.markdown("""
    Welcome to PitGenius! This app predicts Formula 1 pit stop strategies using machine learning.
    Select a race and driver to see predicted pit stop windows and strategy insights.
    """)
    
    try:
        # Load the model
        seasons = [2022, 2023]
        model = load_model(seasons)
        
        # Race selection
        st.sidebar.header('Race Selection')
        year = st.sidebar.selectbox('Select Year', [2024])
        gp_name = st.sidebar.selectbox('Select Grand Prix', ['Bahrain'])
        
        # Team selection
        st.header('Driver Selection')
        teams = sorted(list(set(driver['team'] for driver in drivers_2024)))
        selected_team = st.selectbox('Select Team', teams)
        
        # Filter drivers by selected team
        team_drivers = [driver for driver in drivers_2024 if driver['team'] == selected_team]
        driver_names = [f"{driver['name']} ({driver['code']})" for driver in team_drivers]
        selected_driver_name = st.selectbox('Select Driver', driver_names)
        
        # Extract driver code from selection
        selected_driver_code = selected_driver_name.split('(')[1].split(')')[0]
        
        if st.button('Predict Pit Stops'):
            with st.spinner('Getting race data...'):
                # Get race data
                race_data = get_race_data(year, gp_name)
                
                # Filter for selected driver
                driver_data = race_data[race_data['Driver'] == selected_driver_code].copy()
                
                if not driver_data.empty:
                    # Prepare features
                    features = prepare_features(driver_data)
                    
                    if not features.empty:
                        # Make predictions
                        predictions = predict_race_pit_stops(model, features)
                        
                        # Display results
                        st.subheader('Pit Stop Predictions')
                        
                        # Create visualization
                        fig = plot_driver_prediction(predictions, selected_driver_code)
                        st.pyplot(fig)
                        
                        # Display strategy insights
                        display_strategy_insights(predictions, selected_driver_name)
                    else:
                        st.error("Could not prepare features for prediction. Please try another driver or race.")
                else:
                    st.error(f"No data available for {selected_driver_name} in this race.")
                    
    except Exception as e:
        logger.error(f"Error in Streamlit app: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 