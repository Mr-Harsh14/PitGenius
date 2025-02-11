# Development Guide

This guide outlines the development practices, coding standards, and contribution guidelines for the PitGenius project.

## Getting Started

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/PitGenius.git
   cd PitGenius
   ```

2. **Set Up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Unix/macOS
   pip install -r requirements.txt
   ```

3. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

### Python Style Guide

We follow PEP 8 with some modifications:

1. **Naming Conventions**
   ```python
   # Classes: CamelCase
   class RandomForestModel:
       pass
   
   # Functions and variables: snake_case
   def calculate_pit_window():
       lap_time = 0
   
   # Constants: UPPERCASE
   MAX_LAPS = 50
   ```

2. **Docstrings**
   ```python
   def predict_pit_window(race_data: pd.DataFrame) -> pd.DataFrame:
       """
       Predict optimal pit windows for a race.
       
       Args:
           race_data (pd.DataFrame): DataFrame containing race telemetry
               Required columns:
               - LapNumber (int)
               - TyreLife (int)
               - Position (int)
       
       Returns:
           pd.DataFrame: DataFrame with pit window predictions
               Added columns:
               - OptimalPitWindow (bool)
               - PitWindowScore (float)
       
       Raises:
           ValueError: If required columns are missing
       """
       pass
   ```

3. **Type Hints**
   ```python
   from typing import List, Dict, Optional
   
   def process_lap_data(
       lap_times: List[float],
       driver_info: Dict[str, str],
       weather: Optional[str] = None
   ) -> pd.DataFrame:
       pass
   ```

### Project Structure

1. **Module Organization**
   ```python
   # src/models/random_forest_model.py
   
   from typing import Dict, Optional
   import numpy as np
   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   
   from .base_model import BaseModel
   from ..utils.validation import validate_features
   
   class RandomForestModel(BaseModel):
       """Random Forest implementation for pit stop prediction."""
   ```

2. **Import Order**
   ```python
   # Standard library
   import os
   import json
   from typing import List
   
   # Third-party packages
   import numpy as np
   import pandas as pd
   from sklearn.model_selection import train_test_split
   
   # Local modules
   from .utils import load_config
   from .models import RandomForestModel
   ```

## Testing

### Unit Tests

1. **Test Structure**
   ```python
   # tests/unit/test_random_forest_model.py
   
   import pytest
   import numpy as np
   from src.models import RandomForestModel
   
   @pytest.fixture
   def model():
       return RandomForestModel()
   
   def test_prediction_shape(model):
       X = np.random.rand(100, 10)
       predictions = model.predict(X)
       assert predictions.shape == (100,)
   ```

2. **Running Tests**
   ```bash
   # Run all tests
   pytest
   
   # Run specific test file
   pytest tests/unit/test_random_forest_model.py
   
   # Run with coverage
   pytest --cov=src tests/
   ```

### Integration Tests

```python
# tests/integration/test_prediction_pipeline.py

def test_end_to_end_prediction():
    # Load test data
    race_data = load_test_data()
    
    # Process data
    features = engineer_features(race_data)
    
    # Make predictions
    predictor = PitStopPredictor()
    predictions = predictor.predict(features)
    
    # Validate results
    assert_valid_predictions(predictions)
```

## Git Workflow

1. **Branch Naming**
   - Feature branches: `feature/description`
   - Bug fixes: `fix/description`
   - Documentation: `docs/description`

2. **Commit Messages**
   ```
   feat: Add tire degradation prediction
   
   - Implement new tire degradation model
   - Add validation for tire age
   - Update documentation
   
   Closes #123
   ```

3. **Pull Request Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Changes Made
   - Change 1
   - Change 2
   
   ## Testing
   - [ ] Unit tests added
   - [ ] Integration tests updated
   
   ## Documentation
   - [ ] API docs updated
   - [ ] Comments added
   ```

## Code Review

### Review Checklist

1. **Functionality**
   - [ ] Code works as intended
   - [ ] Edge cases handled
   - [ ] Error handling implemented

2. **Code Quality**
   - [ ] Follows style guide
   - [ ] Well-documented
   - [ ] No duplicate code

3. **Testing**
   - [ ] Tests cover new code
   - [ ] Tests pass
   - [ ] Edge cases tested

## Continuous Integration

### GitHub Actions

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest
```

## Documentation

### API Documentation

1. **Function Documentation**
   ```python
   def calculate_pit_window(
       lap_times: List[float],
       tire_age: int,
       position: int
   ) -> bool:
       """
       Calculate if current lap is optimal for pit stop.
       
       Args:
           lap_times: List of previous lap times
           tire_age: Current tire age in laps
           position: Current race position
       
       Returns:
           bool: True if optimal pit window
       """
       pass
   ```

2. **Class Documentation**
   ```python
   class PitStopPredictor:
       """
       Predicts optimal pit stop windows during races.
       
       Attributes:
           model: Trained prediction model
           scaler: Feature scaler
       
       Methods:
           predict: Make pit stop predictions
           evaluate: Evaluate prediction accuracy
       """
       pass
   ```

## Performance Optimization

1. **Profiling**
   ```python
   import cProfile
   
   def profile_predictions():
       profiler = cProfile.Profile()
       profiler.enable()
       
       # Run predictions
       make_predictions()
       
       profiler.disable()
       profiler.print_stats(sort='cumtime')
   ```

2. **Memory Management**
   ```python
   # Use generators for large datasets
   def process_race_data(file_path):
       with open(file_path) as f:
           for line in f:
               yield process_line(line)
   ```

## Deployment

1. **Environment Variables**
   ```bash
   # .env
   MODEL_VERSION=2023
   CACHE_DIR=data/raw/fastf1_cache
   LOG_LEVEL=INFO
   ```

2. **Logging**
   ```python
   import logging
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       filename='logs/app.log'
   )
   ```

## Support

For development support:
1. Check existing issues
2. Review documentation
3. Ask in team chat
4. Create new issue 