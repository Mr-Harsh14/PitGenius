# Model Architectures Documentation

## Overview
This document provides detailed information about the machine learning model architectures implemented in PitGenius for Formula 1 pit stop prediction. Each model has been designed and optimized for the specific challenges of F1 race strategy prediction.

## Base Model Architecture

### Common Components
```python
class BaseModel:
    def __init__(self, target_variable='has_pit_stop'):
        self.target_variable = target_variable
        self.model = None
        self.feature_names = None
        
    def prepare_data(self):
        # Common data preparation logic
        
    def evaluate(self):
        # Standard evaluation metrics
```

### Shared Functionality
- Data loading and validation
- Train-test splitting
- Performance metric calculation
- Model persistence
- Cross-validation support

## 1. Random Forest Model

### Architecture Overview
```python
class RandomForestModel(BaseModel):
    def __init__(self, target_variable='has_pit_stop'):
        super().__init__(target_variable)
        self.model = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
```

### Hyperparameters
- **n_estimators**: [100, 200, 300]
- **max_depth**: [None, 5, 10]
- **min_samples_split**: [2, 5, 10]
- **class_weight**: 'balanced'
- **random_state**: 42

### Optimization Strategy
1. **Grid Search Parameters**:
   ```python
   param_grid = {
       'n_estimators': [100, 200, 300],
       'max_depth': [None, 5, 10],
       'min_samples_split': [2, 5, 10]
   }
   ```

2. **Cross-Validation**:
   - 5-fold cross-validation
   - Stratified sampling for imbalanced classes
   - F1 score optimization

### Feature Importance Analysis
```python
def get_feature_importance(self):
    importance = self.model.feature_importances_
    return pd.DataFrame({
        'feature': self.feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
```

## 2. Support Vector Machine (SVM)

### Architecture Overview
```python
class SVMModel(BaseModel):
    def __init__(self, target_variable='has_pit_stop'):
        super().__init__(target_variable)
        self.model = SVC(
            class_weight='balanced',
            random_state=42,
            probability=True
        )
```

### Hyperparameters
- **C**: [0.1, 1, 10]
- **kernel**: ['linear', 'rbf', 'poly']
- **gamma**: [0.1, 1, 10]
- **class_weight**: 'balanced'
- **probability**: True

### Optimization Strategy
1. **Grid Search Parameters**:
   ```python
   param_grid = {
       'C': [0.1, 1, 10],
       'kernel': ['linear', 'rbf', 'poly'],
       'gamma': [0.1, 1, 10]
   }
   ```

2. **Cross-Validation**:
   - 5-fold cross-validation
   - Stratified sampling
   - F1 score optimization

### Kernel Selection
- Linear: For linearly separable data
- RBF: For non-linear patterns
- Polynomial: For complex decision boundaries

## 3. Neural Network

### Architecture Overview
```python
class NeuralNetworkModel(BaseModel):
    def create_model(self, input_dim, hidden_layers, dropout_rate, l2_lambda):
        model = Sequential()
        model.add(Dense(hidden_layers[0], input_dim=input_dim,
                       activation='relu',
                       kernel_regularizer=l2(l2_lambda)))
        model.add(Dropout(dropout_rate))
        
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu',
                          kernel_regularizer=l2(l2_lambda)))
            model.add(Dropout(dropout_rate))
            
        model.add(Dense(1, activation='sigmoid'))
        return model
```

### Layer Configurations
1. **Input Layer**:
   - Dynamic input dimension based on features
   - ReLU activation
   - L2 regularization

2. **Hidden Layers**:
   - Multiple configurations: [64, 64], [64, 64, 64], [128, 64, 32]
   - ReLU activation
   - Dropout layers
   - L2 regularization

3. **Output Layer**:
   - Single unit
   - Sigmoid activation
   - Binary classification output

### Hyperparameters
- **hidden_layers**: [[64, 64], [64, 64, 64], [128, 64, 32]]
- **l2_lambda**: [0.0001, 0.0005, 0.001]
- **dropout_rate**: [0.2, 0.3, 0.4]
- **batch_size**: [128, 256, 512]
- **epochs**: [20, 30, 40]

### Training Configuration
```python
model.compile(
    optimizer=Nadam(),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.F1Score()]
)

early_stopping = EarlyStopping(
    monitor='val_f1_score',
    patience=5,
    mode='max',
    restore_best_weights=True
)
```

### Optimization Strategy
1. **Manual Grid Search**:
   - Iterative parameter testing
   - Early stopping monitoring
   - Best model selection based on F1 score

2. **Training Process**:
   - Batch training with validation split
   - Class weight balancing
   - Early stopping on F1 score
   - Best weights restoration

## Model Comparison and Selection

### Evaluation Metrics
1. **Primary Metrics**:
   - F1 Score
   - Precision
   - Recall
   - ROC-AUC

2. **Secondary Metrics**:
   - Training time
   - Inference speed
   - Model size
   - Memory usage

### Selection Criteria
1. **Performance**:
   - F1 score on validation set
   - Generalization capability
   - Class balance handling

2. **Practical Considerations**:
   - Training efficiency
   - Inference speed
   - Resource requirements
   - Maintainability

## Future Architecture Improvements

### 1. Model Enhancements
- Ensemble methods combination
- Advanced regularization techniques
- Dynamic architecture adaptation

### 2. Training Optimizations
- Learning rate scheduling
- Advanced batch selection
- Curriculum learning

### 3. Infrastructure Updates
- Model versioning
- Automated hyperparameter tuning
- Distributed training support 