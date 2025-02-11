# Data Module API Reference

The data module provides functionality for collecting, cleaning, and processing Formula 1 race data. It consists of three main components: data collection, data cleaning, and feature engineering.

## DataCollector

`src.data.data_collector.DataCollector`

The DataCollector class handles fetching and storing raw race data from the FastF1 API.

### Methods

#### `__init__(cache_dir: str = None)`
Initialize the data collector.

**Parameters:**
- `cache_dir`: Optional path to FastF1 cache directory. Defaults to value in `.env`.

#### `collect_race_data(year: int, race_name: str) -> pd.DataFrame`
Collect telemetry and timing data for a specific race.

**Parameters:**
- `year`: The year of the race (e.g., 2023)
- `race_name`: Name of the Grand Prix (e.g., "Abu Dhabi")

**Returns:**
- DataFrame containing combined telemetry and timing data

#### `get_weather_data(session) -> pd.DataFrame`
Extract weather conditions during the race.

**Parameters:**
- `session`: FastF1 session object

**Returns:**
- DataFrame with weather metrics

## DataCleaner

`src.data.data_cleaner.DataCleaner`

The DataCleaner class handles data preprocessing and cleaning operations.

### Methods

#### `__init__()`
Initialize the data cleaner.

#### `clean_race_data(df: pd.DataFrame) -> pd.DataFrame`
Clean and preprocess raw race data.

**Parameters:**
- `df`: Raw race data DataFrame

**Returns:**
- Cleaned DataFrame

#### `handle_missing_values(df: pd.DataFrame) -> pd.DataFrame`
Fill missing values using appropriate strategies.

**Parameters:**
- `df`: DataFrame with missing values

**Returns:**
- DataFrame with handled missing values

#### `remove_outliers(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame`
Remove statistical outliers from specified columns.

**Parameters:**
- `df`: Input DataFrame
- `columns`: List of column names to check for outliers

**Returns:**
- DataFrame with outliers removed

## FeatureEngineering

`src.data.feature_engineering.FeatureEngineering`

The FeatureEngineering class handles creation and transformation of features for the model.

### Methods

#### `__init__()`
Initialize the feature engineering processor.

#### `create_features(df: pd.DataFrame) -> pd.DataFrame`
Generate all features from cleaned race data.

**Parameters:**
- `df`: Cleaned race data DataFrame

**Returns:**
- DataFrame with engineered features

#### `calculate_tire_features(df: pd.DataFrame) -> pd.DataFrame`
Calculate tire-related features.

**Parameters:**
- `df`: Input DataFrame

**Returns:**
- DataFrame with added tire features

#### `calculate_position_features(df: pd.DataFrame) -> pd.DataFrame`
Calculate position and gap-related features.

**Parameters:**
- `df`: Input DataFrame

**Returns:**
- DataFrame with added position features

#### `calculate_strategy_features(df: pd.DataFrame) -> pd.DataFrame`
Calculate strategic opportunity features.

**Parameters:**
- `df`: Input DataFrame

**Returns:**
- DataFrame with added strategy features

## Usage Examples

### Collecting Race Data

```python
from src.data.data_collector import DataCollector

collector = DataCollector()
race_data = collector.collect_race_data(2023, "Abu Dhabi")
```

### Cleaning Data

```python
from src.data.data_cleaner import DataCleaner

cleaner = DataCleaner()
cleaned_data = cleaner.clean_race_data(race_data)
```

### Engineering Features

```python
from src.data.feature_engineering import FeatureEngineering

engineer = FeatureEngineering()
features = engineer.create_features(cleaned_data)
```

## Data Schemas

### Raw Data Schema

Key columns in the raw race data:

```python
{
    'Time': 'datetime64[ns]',       # Timestamp
    'Driver': 'string',             # Driver ID
    'LapTime': 'float64',           # Lap time in seconds
    'Position': 'int64',            # Current position
    'TyreLife': 'int64',           # Current tire age
    'Compound': 'string',           # Tire compound
    'Speed': 'float64',             # Current speed
    'Weather': 'string'             # Weather conditions
}
```

### Feature Schema

Key engineered features:

```python
{
    'TyreLife': 'int64',           # Tire age in laps
    'DegradationPercentage': 'float64', # Tire degradation
    'RollingPaceSeconds': 'float64',    # Average pace
    'PitWindowOptimality': 'float64',   # Pit window score
    'UndercutPotential': 'float64',     # Undercut opportunity
    'TrackEvolution': 'float64',        # Track condition
    'OptimalPitWindow': 'int64'         # Target variable
}
``` 