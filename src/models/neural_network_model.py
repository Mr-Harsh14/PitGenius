import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from src.models.base_model import BaseModel
import numpy as np
import logging
from sklearn.metrics import f1_score, classification_report

logger = logging.getLogger(__name__)

class NeuralNetworkModel(BaseModel):
    def __init__(self, target_variable='has_pit_stop'):
        """Initialize Neural Network model.
        
        Args:
            target_variable (str): Target variable to predict ('has_pit_stop' or 'good_pit_stop')
        """
        super().__init__(target_variable)
        self.scaler = StandardScaler()
        
    def create_model(self, input_dim, hidden_layers, dropout_rate, l2_lambda):
        """Create neural network model with specified architecture.
        
        Args:
            input_dim (int): Number of input features
            hidden_layers (list): List of integers specifying number of units in each hidden layer
            dropout_rate (float): Dropout rate for regularization
            l2_lambda (float): L2 regularization parameter
        """
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            hidden_layers[0], 
            input_dim=input_dim,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
        ))
        model.add(Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(
                units,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
            ))
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Nadam(),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.F1Score()]
        )
        
        return model
        
    def prepare_data(self, use_pca=False):
        """Prepare data for training by splitting and scaling."""
        super().prepare_data(use_pca)
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
    def train(self, custom_param_grid=None):
        """Train Neural Network model with grid search.
        
        Args:
            custom_param_grid (dict, optional): Custom parameter grid for search
        """
        # Default parameter grid based on the paper
        param_grid = {
            'hidden_layers': [[64, 64], [64, 64, 64], [128, 64, 32]],
            'l2_lambda': [0.0001, 0.0005, 0.001],
            'dropout_rate': [0.2, 0.3, 0.4],
            'batch_size': [128, 256, 512],
            'epochs': [20, 30, 40]
        }
        
        # Use custom grid if provided
        if custom_param_grid is not None:
            param_grid = custom_param_grid
            
        logger.info("Starting Neural Network training with grid search...")
        
        best_f1 = 0
        best_params = None
        best_model = None
        
        # Manual grid search since Keras doesn't work well with sklearn's GridSearchCV
        for hidden_layers in param_grid['hidden_layers']:
            for l2_lambda in param_grid['l2_lambda']:
                for dropout_rate in param_grid['dropout_rate']:
                    for batch_size in param_grid['batch_size']:
                        for epochs in param_grid['epochs']:
                            logger.info(f"Trying parameters: {hidden_layers}, {l2_lambda}, {dropout_rate}, {batch_size}, {epochs}")
                            
                            # Create and compile model
                            self.model = self.create_model(
                                input_dim=self.X_train.shape[1],
                                hidden_layers=hidden_layers,
                                dropout_rate=dropout_rate,
                                l2_lambda=l2_lambda
                            )
                            
                            # Early stopping callback
                            early_stopping = EarlyStopping(
                                monitor='val_f1_score',
                                patience=5,
                                mode='max',
                                restore_best_weights=True
                            )
                            
                            # Train model
                            history = self.model.fit(
                                self.X_train,
                                self.y_train,
                                validation_split=0.2,
                                epochs=epochs,
                                batch_size=batch_size,
                                class_weight=self.class_weights,
                                callbacks=[early_stopping],
                                verbose=0
                            )
                            
                            # Evaluate model
                            _, _, f1 = self.model.evaluate(
                                self.X_test,
                                self.y_test,
                                verbose=0
                            )
                            
                            # Update best model if better F1 score
                            if f1 > best_f1:
                                best_f1 = f1
                                best_params = {
                                    'hidden_layers': hidden_layers,
                                    'l2_lambda': l2_lambda,
                                    'dropout_rate': dropout_rate,
                                    'batch_size': batch_size,
                                    'epochs': epochs
                                }
                                best_model = self.model
        
        # Set best model and log results
        self.model = best_model
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best F1 score: {best_f1:.3f}")
        
    def predict(self, X):
        """Make predictions using the trained model.
        
        Args:
            X: Input features to predict on
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions
        return (self.model.predict(X_scaled) > 0.5).astype(int)
        
    def evaluate(self):
        """Evaluate the model on test data."""
        # Make binary predictions
        y_pred = self.predict(self.X_test)
        
        # Calculate F1 score
        f1 = f1_score(self.y_test, y_pred)
        
        # Get detailed classification report
        report = classification_report(self.y_test, y_pred)
        
        logger.info(f"Test set F1 score: {f1:.3f}")
        logger.info(f"Classification Report:\n{report}")
        
        return f1, report 