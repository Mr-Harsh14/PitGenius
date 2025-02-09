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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set the project root as the parent directory of this file
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import project modules
from src.models.train_random_forest import load_model
from src.models.predict_pit_stops import get_race_data, prepare_features, predict_race_pit_stops

# Initialize FastF1 cache in the streamlit static directory
cache_dir = Path('.streamlit/fastf1_cache')
cache_dir.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(cache_dir))

# Page config
st.set_page_config(
    page_title="PitGenius - F1 Strategy Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Title and description
st.title("🏎️ PitGenius: F1 Pit Stop Predictions")
st.markdown("""
Predict pit stop strategies for Formula 1 races using machine learning.
Select a race and driver to see predicted pit stops and tire compounds.
""")

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

@st.cache_resource
def load_trained_model():
    """Load the trained model (cached)"""
    model_path = Path('models/random_forest/rf_model_2022-2023.joblib')
    if not model_path.exists():
        st.error("Model file not found. Please train the model first.")
        return None
    return load_model([2022, 2023])

@st.cache_data
def get_2024_schedule():
    """Get the 2024 F1 race schedule (cached)"""
    schedule = fastf1.get_event_schedule(2024)
    return schedule[schedule['EventFormat'] == 'conventional']['EventName'].tolist()

def main():
    # Sidebar for inputs
    st.sidebar.header("Race Selection")
    
    # Get available races for 2024
    races = get_2024_schedule()
    
    # Race selection
    selected_race = st.sidebar.selectbox(
        "Select Race",
        races
    )
    
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
            model = load_trained_model()
            if model is None:
                return
            
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