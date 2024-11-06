import streamlit as st
from typing import Dict, Any
from datetime import datetime

class Sidebar:
    def __init__(self):
        self.current_year = datetime.now().year
        self.available_years = range(2019, self.current_year + 1)
        
    def render(self) -> Dict[str, Any]:
        """Render sidebar and return selected settings"""
        st.sidebar.header("Race Settings")
        
        # Year and Grand Prix Selection
        year = st.sidebar.selectbox(
            "Select Year",
            options=self.available_years,
            index=len(self.available_years)-1
        )
        
        # We'll need to fetch this based on the selected year
        grand_prix = st.sidebar.selectbox(
            "Select Grand Prix",
            options=["Australian GP", "Bahrain GP", "Chinese GP"]  # This will be dynamic
        )
        
        # Weather Conditions
        weather_conditions = st.sidebar.selectbox(
            "Weather Conditions",
            options=["Dry", "Wet", "Mixed"]
        )
        
        # Track Temperature
        track_temp = st.sidebar.slider(
            "Track Temperature (°C)",
            min_value=15,
            max_value=60,
            value=30
        )
        
        # Tire Compounds Available
        tire_compounds = st.sidebar.multiselect(
            "Available Tire Compounds",
            options=["Soft", "Medium", "Hard", "Intermediate", "Wet"],
            default=["Soft", "Medium", "Hard"]
        )
        
        # Strategy Risk Profile
        risk_profile = st.sidebar.select_slider(
            "Strategy Risk Profile",
            options=["Conservative", "Balanced", "Aggressive"],
            value="Balanced"
        )
        
        return {
            "year": year,
            "grand_prix": grand_prix,
            "weather": weather_conditions,
            "track_temp": track_temp,
            "tire_compounds": tire_compounds,
            "risk_profile": risk_profile
        } 