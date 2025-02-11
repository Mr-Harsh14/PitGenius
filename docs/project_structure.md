# Project Structure

This document outlines the organization and structure of the PitGenius project.

## Directory Layout

```
PitGenius/
├── app/                    # Web application files
├── configs/                # Configuration files
├── data/                   # Data directory
│   ├── raw/               # Raw race data
│   ├── processed/         # Processed features
│   └── validation/        # Validation reports
├── docs/                   # Documentation
│   ├── api/               # API reference
│   ├── user_guides/       # User guides
│   └── development/       # Development guides
├── frontend/              # Frontend components
├── models/                # Trained models
│   └── random_forest/     # Random Forest models
├── notebooks/             # Jupyter notebooks
├── predictions/           # Model predictions
├── reports/               # Analysis reports
│   └── figures/          # Generated graphics
├── scripts/               # Utility scripts
├── src/                   # Source code
│   ├── data/             # Data processing
│   ├── models/           # Model implementations
│   ├── visualization/    # Visualization tools
│   └── utils/            # Utility functions
└── tests/                # Test suite
```

## Key Components

### Source Code (`src/`)

1. **Data Module** (`src/data/`)
   - `data_collector.py`: FastF1 API integration
   - `data_cleaner.py`: Data preprocessing
   - `feature_engineering.py`: Feature creation

2. **Models Module** (`src/models/`)
   - `base_model.py`: Abstract model interface
   - `random_forest_model.py`: Random Forest implementation
   - `neural_network_model.py`: Neural network implementation
   - `predict_pit_stops.py`: Prediction pipeline

3. **Visualization Module** (`src/visualization/`)
   - `pit_stop_viz.py`: Pit stop strategy visualization
   - `plotting.py`: General plotting utilities

4. **Utils Module** (`src/utils/`)
   - Helper functions
   - Common utilities

### Web Application (`app/`)

- Streamlit dashboard
- Interactive visualizations
- Real-time predictions

### Data Management (`data/`)

1. **Raw Data** (`data/raw/`)
   - FastF1 cache
   - Race telemetry
   - Timing data

2. **Processed Data** (`data/processed/`)
   - Engineered features
   - Training datasets
   - Validation sets

3. **Validation Reports** (`data/validation/`)
   - Feature quality reports
   - Data validation results

### Model Artifacts (`models/`)

- Trained model files
- Model metrics
- Feature importance

### Configuration (`configs/`)

```yaml
# Model Configuration
model:
  type: random_forest
  params:
    n_estimators: 100
    max_depth: 10
    random_state: 42

# Data Configuration
data:
  cache_dir: data/raw/fastf1_cache
  features_dir: data/processed/features

# Training Configuration
training:
  test_size: 0.2
  random_state: 42
  cross_validation: 5
```

### Scripts (`scripts/`)

1. **Data Collection**
   ```bash
   collect_race_data.py --year 2023 --race "Abu Dhabi"
   ```

2. **Feature Engineering**
   ```bash
   engineer_features.py --race "abu_dhabi_grand_prix"
   ```

3. **Model Training**
   ```bash
   train_model.py --model random_forest --data training_data.csv
   ```

4. **Prediction**
   ```bash
   predict_pit_stops.py --race "abu_dhabi_grand_prix"
   ```

## Development Tools

### Environment Management

1. **Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Testing

1. **Unit Tests**
   ```bash
   pytest tests/unit/
   ```

2. **Integration Tests**
   ```bash
   pytest tests/integration/
   ```

### Documentation

1. **API Documentation**
   - Module references
   - Class documentation
   - Function signatures

2. **User Guides**
   - Installation guide
   - Usage tutorials
   - Best practices

## File Naming Conventions

1. **Python Files**
   - Lowercase with underscores
   - Descriptive names
   - Example: `feature_engineering.py`

2. **Class Names**
   - CapWords convention
   - Example: `RandomForestModel`

3. **Test Files**
   - Prefix with `test_`
   - Example: `test_data_cleaner.py`

## Version Control

1. **Git Structure**
   - `main`: Production-ready code
   - `develop`: Development branch
   - Feature branches: `feature/new-feature`
   - Hotfix branches: `hotfix/bug-fix`

2. **.gitignore**
   - Excludes cache files
   - Ignores large data files
   - Skips environment files

## Deployment

1. **Production Setup**
   - Environment configuration
   - Dependency management
   - Logging setup

2. **Monitoring**
   - Performance metrics
   - Error tracking
   - Usage statistics 