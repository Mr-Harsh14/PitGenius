import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import fastf1
import fastf1.plotting
from datetime import datetime
from matplotlib.lines import Line2D

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_race_data(year: int, gp_name: str) -> tuple:
    """
    Load race data and predictions.
    
    Args:
        year: Season year
        gp_name: Name of the Grand Prix
        
    Returns:
        tuple: (race data, predictions)
    """
    # Load race data
    cache_dir = Path(project_root) / 'data' / 'raw' / 'fastf1_cache'
    fastf1.Cache.enable_cache(str(cache_dir))
    
    race = fastf1.get_session(year, gp_name, 'R')
    race.load()
    
    # Load predictions
    race_name = gp_name.lower().replace(' ', '_')
    pred_path = Path(project_root) / 'predictions' / str(year) / f'{race_name}_predictions.csv'
    predictions = pd.read_csv(pred_path)
    
    return race, predictions

def plot_driver_pit_stops(race, predictions: pd.DataFrame, driver_number: str):
    """
    Create visualization of actual vs predicted pit stops for a driver.
    
    Args:
        race: FastF1 race session
        predictions: DataFrame with pit stop predictions
        driver_number: Driver's number to visualize
    """
    # Get driver's laps
    driver_laps = race.laps.pick_driver(driver_number)
    driver_info = race.get_driver(driver_number)
    
    # Get actual pit stops
    pit_stops = driver_laps[driver_laps['PitOutTime'].notna()]['LapNumber'].values
    
    # Filter predictions for this driver
    driver_predictions = predictions[
        predictions['Driver'] == driver_info['Abbreviation']
    ].sort_values('LapNumber')
    
    if driver_predictions.empty:
        logger.warning(f"No predictions found for driver {driver_info['Abbreviation']}")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1], gridspec_kw={'hspace': 0.1})
    
    # Plot pit stop probability on top subplot
    ax1.plot(driver_predictions['LapNumber'], driver_predictions['PitProbability'], 
             label='Pit Stop Probability', color='blue', alpha=0.8, linewidth=2)
    
    # Plot prediction threshold
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, 
                label='Prediction Threshold')
    
    # Plot actual pit stops
    for pit_lap in pit_stops:
        ax1.axvline(x=pit_lap, color='red', linestyle='-', alpha=0.7, linewidth=2)
    
    # Plot predicted pit stops
    predicted_stops = driver_predictions[driver_predictions['PredictedPitStop']]['LapNumber']
    for pred_lap in predicted_stops:
        ax1.axvline(x=pred_lap, color='green', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add legend to top subplot
    legend_elements = [
        Line2D([0], [0], color='blue', label='Pit Stop Probability', alpha=0.8, linewidth=2),
        Line2D([0], [0], color='gray', linestyle='--', label='Prediction Threshold', alpha=0.5),
        Line2D([0], [0], color='red', label='Actual Pit Stop', alpha=0.7, linewidth=2),
        Line2D([0], [0], color='green', linestyle='--', label='Predicted Pit Stop', alpha=0.5, linewidth=2)
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Customize top subplot
    ax1.set_title(f'Pit Stop Predictions vs Actual - {driver_info["FullName"]}', pad=20)
    ax1.set_ylabel('Pit Stop Probability')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(driver_laps['LapNumber']) + 1)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot tyre compounds on bottom subplot
    compounds = {'SOFT': 'salmon', 'MEDIUM': 'yellow', 'HARD': 'white', 'INTERMEDIATE': 'lightgreen', 'WET': 'lightblue'}
    
    # Create separate axes for actual and predicted compounds
    ax2.set_ylim(0, 1)
    
    # Plot actual compounds (top half)
    current_compound = None
    start_lap = 0
    
    for idx, lap in driver_laps.iterrows():
        if lap['Compound'] != current_compound:
            if current_compound is not None:
                ax2.axvspan(start_lap, lap['LapNumber'], 
                           ymin=0.5, ymax=1.0,
                           color=compounds.get(current_compound, 'gray'), alpha=0.3)
                # Add compound label
                mid_lap = (start_lap + lap['LapNumber']) / 2
                ax2.text(mid_lap, 0.75, current_compound[0], 
                        horizontalalignment='center', verticalalignment='center')
            current_compound = lap['Compound']
            start_lap = lap['LapNumber']
    
    # Plot the last actual compound
    if current_compound is not None:
        ax2.axvspan(start_lap, max(driver_laps['LapNumber']), 
                   ymin=0.5, ymax=1.0,
                   color=compounds.get(current_compound, 'gray'), alpha=0.3)
        mid_lap = (start_lap + max(driver_laps['LapNumber'])) / 2
        ax2.text(mid_lap, 0.75, current_compound[0], 
                horizontalalignment='center', verticalalignment='center')
    
    # Plot predicted compounds (bottom half)
    current_compound = None
    start_lap = 0
    
    for idx, lap in driver_predictions.iterrows():
        if lap['CurrentCompound'] != current_compound:
            if current_compound is not None:
                ax2.axvspan(start_lap, lap['LapNumber'], 
                           ymin=0.0, ymax=0.5,
                           color=compounds.get(current_compound, 'gray'), alpha=0.3)
                # Add compound label for predicted stint
                mid_lap = (start_lap + lap['LapNumber']) / 2
                ax2.text(mid_lap, 0.25, current_compound[0], color='green',
                        horizontalalignment='center', verticalalignment='center')
            current_compound = lap['CurrentCompound']
            start_lap = lap['LapNumber']
    
    # Plot the last predicted compound
    if current_compound is not None:
        ax2.axvspan(start_lap, max(driver_predictions['LapNumber']), 
                   ymin=0.0, ymax=0.5,
                   color=compounds.get(current_compound, 'gray'), alpha=0.3)
        mid_lap = (start_lap + max(driver_predictions['LapNumber'])) / 2
        ax2.text(mid_lap, 0.25, current_compound[0], color='green',
                horizontalalignment='center', verticalalignment='center')
    
    # Plot predicted pit stops on compound subplot
    for pit_lap in predicted_stops:
        ax2.axvline(x=pit_lap, color='green', linestyle='--', alpha=0.5)
    
    # Plot actual pit stops on compound subplot
    for pit_lap in pit_stops:
        ax2.axvline(x=pit_lap, color='red', linestyle='-', alpha=0.7)
    
    # Customize compound subplot
    ax2.set_xlabel('Lap Number')
    ax2.set_ylabel('Tyre Compound')
    ax2.set_yticks([0.25, 0.75], ['Predicted', 'Actual'])
    ax2.set_xlim(0, max(driver_laps['LapNumber']) + 1)
    
    # Add compound legend
    compound_patches = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.3, label=compound)
                       for compound, color in compounds.items()]
    ax2.legend(handles=compound_patches, loc='upper right', ncol=5)
    
    # Create race-specific directory for plots
    race_name = race.event.EventName.lower().replace(' ', '_')
    plots_dir = Path(project_root) / 'reports' / 'figures' / race_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / f'pit_stops_{driver_info["Abbreviation"]}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_race_overview(race, predictions: pd.DataFrame):
    """
    Create a race overview visualization showing all drivers' pit stops.
    
    Args:
        race: FastF1 race session
        predictions: DataFrame with pit stop predictions
    """
    # Get all drivers
    drivers = race.drivers
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot each driver's pit stops
    for i, driver in enumerate(drivers):
        driver_info = race.get_driver(driver)
        driver_laps = race.laps.pick_driver(driver)
        
        # Add starting compound (actual)
        starting_compound = driver_laps.iloc[0]['Compound']
        plt.text(0, i+0.3, starting_compound[0], 
                color='red', ha='right', va='bottom', fontsize=8)
        
        # Add starting compound (predicted)
        driver_preds = predictions[predictions['Driver'] == driver_info['Abbreviation']]
        if not driver_preds.empty:
            predicted_start = driver_preds.iloc[0]['CurrentCompound']
            plt.text(0, i-0.3, predicted_start[0], 
                    color='green', ha='right', va='top', fontsize=8)
        
        # Plot actual pit stops with compound changes
        pit_stops = driver_laps[driver_laps['PitOutTime'].notna()]
        for _, pit_stop in pit_stops.iterrows():
            plt.plot([pit_stop['LapNumber'], pit_stop['LapNumber']], 
                    [i-0.2, i+0.2], 'r-', linewidth=2, alpha=0.7)
            # Add compound label
            plt.text(pit_stop['LapNumber'], i+0.3, pit_stop['Compound'][0], 
                    color='red', ha='center', va='bottom', fontsize=8)
        
        # Plot predicted pit stops with predicted compounds
        driver_preds = predictions[
            (predictions['Driver'] == driver_info['Abbreviation']) & 
            predictions['PredictedPitStop']
        ]
        
        for _, pred in driver_preds.iterrows():
            plt.plot([pred['LapNumber'], pred['LapNumber']], 
                    [i-0.2, i+0.2], 'g--', alpha=0.5, linewidth=2)
            # Add predicted compound label
            if 'CurrentCompound' in pred:
                next_compound = predictions.loc[
                    (predictions['Driver'] == driver_info['Abbreviation']) & 
                    (predictions['LapNumber'] > pred['LapNumber'])
                ].iloc[0]['CurrentCompound']
                plt.text(pred['LapNumber'], i-0.3, next_compound[0], 
                        color='green', ha='center', va='top', fontsize=8)
    
    # Customize plot
    plt.yticks(range(len(drivers)), 
               [race.get_driver(d)['Abbreviation'] for d in drivers])
    plt.title('Race Pit Stop Overview - Actual vs Predicted', pad=20)
    plt.xlabel('Lap Number')
    plt.ylabel('Driver')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis limits with padding for starting compounds
    plt.xlim(-2, max(race.laps['LapNumber']) + 1)
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], color='red', label='Actual Pit Stop', alpha=0.7, linewidth=2),
        Line2D([0], [0], color='green', linestyle='--', label='Predicted Pit Stop', alpha=0.5, linewidth=2)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Add minor grid
    plt.grid(True, which='minor', alpha=0.15)
    plt.minorticks_on()
    
    # Create race-specific directory for plots
    race_name = race.event.EventName.lower().replace(' ', '_')
    plots_dir = Path(project_root) / 'reports' / 'figures' / race_name
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / 'race_pit_stops_overview.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to create pit stop visualizations for all 2024 races."""
    try:
        # Get 2024 race schedule
        schedule = fastf1.get_event_schedule(2024)
        races = schedule[schedule['EventFormat'] == 'conventional']
        
        # Process each race
        for _, race_event in races.iterrows():
            race_name = race_event['EventName']
            logger.info(f"\nProcessing visualizations for {race_name}")
            
            try:
                # Load race data and predictions
                race, predictions = load_race_data(2024, race_name)
                logger.info("Loaded race data and predictions")
                
                # Create individual driver plots for all drivers
                for driver_number in race.drivers:
                    try:
                        plot_driver_pit_stops(race, predictions, driver_number)
                        logger.info(f"Created visualization for driver {driver_number}")
                    except Exception as e:
                        logger.error(f"Error creating visualization for driver {driver_number}: {str(e)}")
                        continue
                
                # Create race overview
                plot_race_overview(race, predictions)
                logger.info("Created race overview visualization")
                
                logger.info(f"Completed visualizations for {race_name}")
                
            except Exception as e:
                logger.error(f"Error processing visualizations for {race_name}: {str(e)}")
                continue
        
        logger.info("\nAll visualizations completed successfully")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        raise

if __name__ == "__main__":
    main() 