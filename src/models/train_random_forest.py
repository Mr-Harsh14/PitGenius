import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the project root directory to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.random_forest_model import RandomForestModel
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_random_forest(seasons: list, use_pca: bool = False, custom_param_grid: dict = None) -> tuple:
    """
    Train and evaluate Random Forest model for pit stop prediction.
    
    Args:
        seasons: List of seasons to use for training
        use_pca: Whether to use PCA features
        custom_param_grid: Optional custom hyperparameter grid
        
    Returns:
        tuple: (trained model, evaluation results, feature importance)
    """
    logger.info(f"Training Random Forest model for seasons: {seasons}")
    
    # Initialize model
    model = RandomForestModel(target_variable='has_pit_stop')
    
    # Load and combine data from all seasons
    all_data = []
    for season in seasons:
        model.load_data(season)
        if hasattr(model, 'features_df'):
            all_data.append(model.features_df)
    
    if not all_data:
        raise ValueError("No data loaded for any season")
    
    # Combine all seasons' data
    model.features_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data size: {len(model.features_df)} rows")
    
    # Prepare combined data
    model.prepare_data(use_pca=use_pca)
    
    # Train model with optional custom parameters
    model.train(custom_param_grid)
    
    # Evaluate model
    f1_score, classification_report = model.evaluate()
    
    # Get feature importance
    feature_importance = model.get_feature_importance()
    
    logger.info(f"\nModel Performance:")
    logger.info(f"F1 Score: {f1_score:.3f}")
    logger.info(f"\nClassification Report:\n{classification_report}")
    logger.info(f"\nTop 10 Important Features:")
    logger.info(feature_importance.head(10))
    
    return model, {'f1_score': f1_score, 'report': classification_report}, feature_importance

def save_model(model, seasons: list, metrics: dict, feature_importance: pd.DataFrame):
    """
    Save the trained model and its associated metrics.
    
    Args:
        model: Trained Random Forest model
        seasons: List of seasons used for training
        metrics: Dictionary containing model metrics
        feature_importance: DataFrame of feature importance scores
    """
    # Create models directory if it doesn't exist
    models_dir = Path(project_root) / 'models' / 'random_forest'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create season identifier
    season_str = f"{min(seasons)}-{max(seasons)}"
    
    # Save model
    model_path = models_dir / f'rf_model_{season_str}.joblib'
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = models_dir / f'rf_metrics_{season_str}.csv'
    pd.DataFrame([metrics]).to_csv(metrics_path)
    
    # Save feature importance
    importance_path = models_dir / f'rf_feature_importance_{season_str}.csv'
    feature_importance.to_csv(importance_path)
    
    logger.info(f"\nModel and metrics saved in {models_dir}")

def load_model(seasons: list) -> RandomForestModel:
    """
    Load a trained Random Forest model.
    
    Args:
        seasons: List of seasons used in training
        
    Returns:
        RandomForestModel: Loaded model
    """
    season_str = f"{min(seasons)}-{max(seasons)}"
    model_path = Path(project_root) / 'models' / 'random_forest' / f'rf_model_{season_str}.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"No model found for seasons {season_str}")
    
    return joblib.load(model_path)

def predict_pit_stops(model: RandomForestModel, race_data: pd.DataFrame) -> np.ndarray:
    """
    Make pit stop predictions for race data.
    
    Args:
        model: Trained Random Forest model
        race_data: DataFrame containing race features
        
    Returns:
        np.ndarray: Predicted pit stop probabilities
    """
    return model.model.predict_proba(race_data)[:, 1]

def main():
    """Main function to train and save the Random Forest model."""
    # Training configuration
    seasons = [2022, 2023]  # Use both 2022 and 2023 seasons
    use_pca = False
    
    # Optional: Custom parameter grid
    custom_params = {
        'n_estimators': [200, 300, 400],
        'max_depth': [None, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    try:
        # Train model
        model, metrics, importance = train_random_forest(
            seasons=seasons,
            use_pca=use_pca,
            custom_param_grid=custom_params
        )
        
        # Save model and results
        save_model(model, seasons, metrics, importance)
        
        logger.info("Random Forest model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 