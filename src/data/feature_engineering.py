import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class F1FeatureEngineer:
    """Class to engineer features from Formula 1 race data."""
    
    def __init__(self, processed_data_dir: str, interim_data_dir: str):
        """
        Initialize the feature engineer.
        
        Args:
            processed_data_dir: Directory containing cleaned data
            interim_data_dir: Directory to save engineered features
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.interim_data_dir = Path(interim_data_dir)
        self.interim_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Define feature groups
        self.categorical_features = ['Team', 'Driver', 'Compound']
        self.numeric_features = [
            'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST',
            'TyreLife', 'Position', 'QualyPosition', 'DriverStandings'
        ]
        self.time_features = [
            'Time', 'LapTime', 'PitOutTime', 'PitInTime',
            'Sector1Time', 'Sector2Time', 'Sector3Time',
            'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
            'LapStartTime'
        ]
        
        # Initialize transformers
        self.numeric_transformer = StandardScaler()
        self.categorical_transformer = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore'
        )
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
        # Create target variables
        self.target_variables = ['has_pit_stop', 'good_pit_stop']
        
        # Initialize column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ],
            remainder='drop'
        )
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load cleaned race data."""
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def preprocess_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert datetime features to seconds since midnight.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with processed datetime features
        """
        df_copy = df.copy()
        
        for col in self.time_features:
            if col in df_copy.columns:
                # Convert to timedelta if not already
                if not pd.api.types.is_timedelta64_dtype(df_copy[col]):
                    df_copy[col] = pd.to_timedelta(df_copy[col])
                
                # Convert to seconds
                df_copy[f'{col}_seconds'] = df_copy[col].dt.total_seconds()
                
                # Drop original column
                df_copy.drop(col, axis=1, inplace=True)
        
        return df_copy
    
    def preprocess_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert time features to seconds and handle missing values.
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with processed time features
        """
        df_copy = df.copy()
        
        # Get all timedelta columns
        timedelta_cols = [col for col in df_copy.columns 
                         if pd.api.types.is_timedelta64_dtype(df_copy[col].dtype)]
        
        for col in timedelta_cols:
            # Convert to seconds
            df_copy[f'{col}_seconds'] = df_copy[col].dt.total_seconds()
            
            # Drop original column
            df_copy.drop(col, axis=1, inplace=True)
            
            # Handle missing values for pit times
            if col in ['PitInTime', 'PitOutTime']:
                df_copy[f'{col}_seconds'] = df_copy[f'{col}_seconds'].fillna(0)
        
        return df_copy
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for pit stop prediction.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with target variables
        """
        df_copy = df.copy()
        
        # Create has_pit_stop target (1 if PitInTime is not null)
        # Only use PitInTime to avoid future information leakage
        df_copy['has_pit_stop'] = (df_copy['PitInTime_seconds'] > 0).astype(int)
        
        # Create good_pit_stop target using rolling median of previous laps
        df_copy['good_pit_stop'] = 0  # Initialize
        
        # Sort by race and calculate rolling median pit times
        df_copy = df_copy.sort_values(['Year', 'RaceNumber', 'LapNumber'])
        
        # Calculate rolling median for each race
        medians = []
        for (year, race), group in df_copy.groupby(['Year', 'RaceNumber']):
            pit_times = group[group['PitInTime_seconds'] > 0]['PitInTime_seconds']
            if not pit_times.empty:
                median = pit_times.median()
                medians.append((year, race, median))
        
        # Create a lookup dictionary for medians
        median_dict = {(year, race): median for year, race, median in medians}
        
        # Mark pit stops as good if they're faster than their race's median
        mask = df_copy['has_pit_stop'] == 1
        for (year, race), group in df_copy[mask].groupby(['Year', 'RaceNumber']):
            if (year, race) in median_dict:
                median = median_dict[(year, race)]
                idx = group.index
                df_copy.loc[idx, 'good_pit_stop'] = (
                    df_copy.loc[idx, 'PitInTime_seconds'] < median
                ).astype(int)
        
        return df_copy
    
    def apply_feature_engineering(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply feature engineering transformations.
        
        Returns:
            Tuple containing:
            - DataFrame with all engineered features
            - DataFrame with PCA-reduced features
        """
        try:
            logger.info("Starting feature engineering")
            
            # Sort by race and lap number to ensure proper temporal order
            df = df.sort_values(['Year', 'RaceNumber', 'LapNumber'])
            
            # Process time-based features first
            df = self.preprocess_time_features(df)
            df = self.preprocess_datetime_features(df)
            
            # Create lagged features to avoid using future information
            lagged_features = []
            for col in self.numeric_features:
                if col in df.columns:
                    df[f'{col}_prev_lap'] = df.groupby(['Year', 'RaceNumber', 'Driver'])[col].shift(1)
                    lagged_features.append(f'{col}_prev_lap')
            
            # Update numeric features to use lagged versions and available time features
            numeric_features = []
            
            # Add lagged features that exist
            numeric_features.extend([f for f in lagged_features if f in df.columns])
            
            # Add time features that exist (avoiding pit times)
            time_features = [
                col for col in df.columns 
                if col.endswith('_seconds') 
                and not col.startswith(('PitIn', 'PitOut'))
            ]
            numeric_features.extend(time_features)
            
            # Update the preprocessor with available features
            available_cat_features = [f for f in self.categorical_features if f in df.columns]
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', self.numeric_transformer, numeric_features),
                    ('cat', self.categorical_transformer, available_cat_features)
                ],
                remainder='drop'
            )
            
            # Create target variables first
            df_with_targets = self.create_target_variables(df)
            target_values = df_with_targets[self.target_variables]
            
            # Fit and transform the data
            transformed_features = self.preprocessor.fit_transform(df)
            
            # Get feature names
            feature_names = self.preprocessor.get_feature_names_out()
            
            # Create DataFrame with transformed features
            transformed_df = pd.DataFrame(
                transformed_features,
                columns=feature_names,
                index=df.index
            )
            
            # Handle any remaining NaN values before PCA
            if transformed_df.isna().any().any():
                logger.info("Handling remaining NaN values before PCA")
                transformed_df = transformed_df.fillna(0)
            
            # Apply PCA to the transformed features
            pca_features = self.pca.fit_transform(transformed_df)
            n_components = pca_features.shape[1]
            
            pca_df = pd.DataFrame(
                pca_features,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=df.index
            )
            
            # Add target variables to both DataFrames
            for target in self.target_variables:
                transformed_df[target] = target_values[target]
                pca_df[target] = target_values[target]
            
            logger.info(f"Created {len(feature_names)} features, reduced to {n_components} principal components")
            logger.info(f"Explained variance ratio: {self.pca.explained_variance_ratio_.cumsum()[-1]:.3f}")
            
            return transformed_df, pca_df
            
        except Exception as e:
            logger.error(f"Error in feature engineering: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def process_all_files(self) -> None:
        """Process all cleaned data files and save engineered features."""
        try:
            # Get all parquet files
            data_files = list(self.processed_data_dir.glob("cleaned_*.parquet"))
            
            if not data_files:
                logger.warning(f"No cleaned data files found in {self.processed_data_dir}")
                return
            
            for file_path in data_files:
                logger.info(f"Processing {file_path}")
                
                # Load data
                df = self.load_data(file_path)
                if df.empty:
                    continue
                
                # Apply feature engineering
                transformed_df, pca_df = self.apply_feature_engineering(df)
                
                if not transformed_df.empty:
                    # Save transformed features
                    output_base = file_path.stem.replace('cleaned_', '')
                    
                    # Save full feature set
                    full_features_path = self.interim_data_dir / f"features_{output_base}.parquet"
                    transformed_df.to_parquet(full_features_path)
                    logger.info(f"Saved full feature set to {full_features_path}")
                    
                    # Save PCA features
                    pca_features_path = self.interim_data_dir / f"pca_features_{output_base}.parquet"
                    pca_df.to_parquet(pca_features_path)
                    logger.info(f"Saved PCA features to {pca_features_path}")
                
        except Exception as e:
            logger.error(f"Error processing files: {str(e)}")

def main():
    """Main function to run the feature engineering process."""
    # Set up directories
    processed_data_dir = "data/processed"
    interim_data_dir = "data/interim"
    
    # Initialize engineer
    engineer = F1FeatureEngineer(processed_data_dir, interim_data_dir)
    
    # Run feature engineering
    engineer.process_all_files()

if __name__ == "__main__":
    main() 