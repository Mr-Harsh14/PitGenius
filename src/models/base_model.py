import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report
import logging

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, target_variable='has_pit_stop'):
        """Initialize the base model.
        
        Args:
            target_variable (str): Target variable to predict ('has_pit_stop' or 'good_pit_stop')
        """
        self.target_variable = target_variable
        self.model = None
        self.feature_names = None
        
    def load_data(self, season_year):
        """Load the feature engineered data for a specific season.
        
        Args:
            season_year (int): Year of the season to load
        """
        # Load both full features and PCA features
        self.features_df = pd.read_parquet(f'data/interim/features_season_{season_year}.parquet')
        self.pca_features_df = pd.read_parquet(f'data/interim/pca_features_season_{season_year}.parquet')
        
        # Store feature names
        self.feature_names = [col for col in self.features_df.columns 
                            if col not in [self.target_variable, 'good_pit_stop', 'has_pit_stop']]
        
    def prepare_data(self, use_pca=False):
        """Prepare data for training by splitting into train/test sets.
        
        Args:
            use_pca (bool): Whether to use PCA features instead of full features
        """
        if use_pca:
            X = self.pca_features_df
        else:
            X = self.features_df[self.feature_names]
            
        y = self.features_df[self.target_variable]
        
        # Split data 70/30
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Calculate class weights
        class_counts = np.bincount(self.y_train)
        total_samples = len(self.y_train)
        self.class_weights = {
            i: total_samples / (len(class_counts) * count) 
            for i, count in enumerate(class_counts)
        }
        
        logger.info(f"Training set size: {len(self.X_train)}")
        logger.info(f"Test set size: {len(self.X_test)}")
        logger.info(f"Class weights: {self.class_weights}")
        
    def train(self, param_grid):
        """Train the model using grid search and cross validation.
        
        Args:
            param_grid (dict): Grid of hyperparameters to search
        """
        if self.model is None:
            raise ValueError("Model not initialized. Please implement in child class.")
            
        # Setup grid search with 5-fold cross validation
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        
        # Fit grid search
        grid_search.fit(self.X_train, self.y_train)
        
        # Update model with best estimator
        self.model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation F1 score: {grid_search.best_score_:.3f}")
        
    def evaluate(self):
        """Evaluate the model on test data."""
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate F1 score
        f1 = f1_score(self.y_test, y_pred)
        
        # Get detailed classification report
        report = classification_report(self.y_test, y_pred)
        
        logger.info(f"Test set F1 score: {f1:.3f}")
        logger.info(f"Classification Report:\n{report}")
        
        return f1, report 