from sklearn.svm import SVC
from src.models.base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class SVMModel(BaseModel):
    def __init__(self, target_variable='has_pit_stop'):
        """Initialize SVM model.
        
        Args:
            target_variable (str): Target variable to predict ('has_pit_stop' or 'good_pit_stop')
        """
        super().__init__(target_variable)
        
        # Initialize model with class weights
        self.model = SVC(
            class_weight='balanced',
            random_state=42,
            probability=True  # Enable probability estimates
        )
        
    def train(self, custom_param_grid=None):
        """Train SVM model with grid search.
        
        Args:
            custom_param_grid (dict, optional): Custom parameter grid for search
        """
        # Default parameter grid based on the paper
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': [0.1, 1, 10]
        }
        
        # Use custom grid if provided
        if custom_param_grid is not None:
            param_grid = custom_param_grid
            
        logger.info("Starting SVM training with grid search...")
        super().train(param_grid) 