# Models Module API Reference

The models module implements various machine learning models for pit stop strategy prediction. It includes base classes, model implementations, training utilities, and prediction functionality.

## Base Model

`src.models.base_model.BaseModel`

Abstract base class defining the interface for all pit stop strategy models.

### Methods

#### `__init__(model_params: Dict = None)`
Initialize the base model.

**Parameters:**
- `model_params`: Dictionary of model hyperparameters

#### `train(X: np.ndarray, y: np.ndarray) -> None`
Train the model on provided data.

**Parameters:**
- `X`: Feature matrix
- `y`: Target labels

#### `predict(X: np.ndarray) -> np.ndarray`
Make predictions on new data.

**Parameters:**
- `X`: Feature matrix

**Returns:**
- Array of predictions

#### `save_model(path: str) -> None`
Save model to disk.

**Parameters:**
- `path`: Path to save model

#### `load_model(path: str) -> None`
Load model from disk.

**Parameters:**
- `path`: Path to saved model

## Random Forest Model

`src.models.random_forest_model.RandomForestModel`

Random Forest implementation for pit stop strategy prediction.

### Methods

#### `__init__(model_params: Dict = None)`
Initialize the Random Forest model.

**Parameters:**
- `model_params`: Dictionary with keys:
  - `n_estimators`: Number of trees (default: 100)
  - `max_depth`: Maximum tree depth (default: None)
  - `min_samples_split`: Minimum samples for split (default: 2)
  - `random_state`: Random seed (default: 42)

#### `feature_importance(feature_names: List[str]) -> pd.DataFrame`
Get feature importance scores.

**Parameters:**
- `feature_names`: List of feature names

**Returns:**
- DataFrame with feature importance scores

## Neural Network Model

`src.models.neural_network_model.NeuralNetworkModel`

Deep learning model for pit stop strategy prediction.

### Methods

#### `__init__(model_params: Dict = None)`
Initialize the Neural Network model.

**Parameters:**
- `model_params`: Dictionary with keys:
  - `hidden_layers`: List of layer sizes
  - `learning_rate`: Learning rate
  - `batch_size`: Training batch size
  - `epochs`: Number of training epochs

#### `build_model(input_shape: Tuple[int]) -> None`
Build the neural network architecture.

**Parameters:**
- `input_shape`: Shape of input features

## Model Training

`src.models.train_models.ModelTrainer`

Utility class for training and evaluating models.

### Methods

#### `__init__(model: BaseModel, data_path: str)`
Initialize the model trainer.

**Parameters:**
- `model`: Instance of a model class
- `data_path`: Path to training data

#### `train_and_evaluate(test_size: float = 0.2) -> Dict`
Train model and evaluate performance.

**Parameters:**
- `test_size`: Fraction of data for testing

**Returns:**
- Dictionary of evaluation metrics

## Prediction System

`src.models.predict_pit_stops.PitStopPredictor`

System for making pit stop strategy predictions during races.

### Methods

#### `__init__(model_path: str)`
Initialize the prediction system.

**Parameters:**
- `model_path`: Path to trained model

#### `predict_pit_windows(race_data: pd.DataFrame) -> pd.DataFrame`
Predict optimal pit windows for a race.

**Parameters:**
- `race_data`: DataFrame with race features

**Returns:**
- DataFrame with pit window predictions

## Usage Examples

### Training a Random Forest Model

```python
from src.models.random_forest_model import RandomForestModel
from src.models.train_models import ModelTrainer

# Initialize model
model = RandomForestModel({
    'n_estimators': 200,
    'max_depth': 10
})

# Train and evaluate
trainer = ModelTrainer(model, 'data/processed/training_data.csv')
metrics = trainer.train_and_evaluate()
```

### Making Predictions

```python
from src.models.predict_pit_stops import PitStopPredictor

predictor = PitStopPredictor('models/random_forest/model.joblib')
predictions = predictor.predict_pit_windows(race_data)
```

## Model Performance Metrics

The following metrics are used to evaluate model performance:

```python
{
    'accuracy': float,          # Overall accuracy
    'precision': float,         # Precision for pit window prediction
    'recall': float,           # Recall for pit window prediction
    'f1_score': float,         # F1 score
    'confusion_matrix': np.ndarray  # Confusion matrix
}
```

## Model Parameters

### Random Forest Default Parameters

```python
{
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'random_state': 42
}
```

### Neural Network Default Parameters

```python
{
    'hidden_layers': [64, 32, 16],
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100
}
``` 