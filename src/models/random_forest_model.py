from sklearn.ensemble import RandomForestClassifier
from src.models.base_model import BaseModel
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class RandomForestModel(BaseModel):
    def __init__(self, target_variable='has_pit_stop'):
        """Initialize Random Forest model.
        
        Args:
            target_variable (str): Target variable to predict ('has_pit_stop' or 'good_pit_stop')
        """
        super().__init__(target_variable)
        
        # Initialize model with class weights
        self.model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
    def train(self, custom_param_grid=None):
        """Train Random Forest model with grid search.
        
        Args:
            custom_param_grid (dict, optional): Custom parameter grid for search
        """
        # Default parameter grid based on the paper
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        
        # Use custom grid if provided
        if custom_param_grid is not None:
            param_grid = custom_param_grid
            
        logger.info("Starting Random Forest training with grid search...")
        super().train(param_grid)
        
    def get_feature_importance(self):
        """Get feature importance scores from the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Please train the model first.")
            
        # Get feature importance scores
        importance = self.model.feature_importances_
        
        # Create DataFrame with feature names and importance scores
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # Sort by importance
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        return feature_importance 