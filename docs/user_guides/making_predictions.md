# Making Pit Stop Predictions

This guide explains how to use PitGenius to make pit stop strategy predictions for Formula 1 races.

## Quick Start

1. **Collect Race Data**
   ```bash
   python scripts/collect_race_data.py --year 2023 --race "Abu Dhabi"
   ```

2. **Generate Features**
   ```bash
   python scripts/engineer_features.py --race "abu_dhabi_grand_prix"
   ```

3. **Make Predictions**
   ```bash
   python scripts/predict_pit_stops.py --race "abu_dhabi_grand_prix"
   ```

4. **View Results**
   ```bash
   streamlit run frontend/pit_stop_app.py
   ```

## Detailed Steps

### 1. Data Collection

The first step is collecting race data using the FastF1 API:

```bash
python scripts/collect_race_data.py --year 2023 --race "Abu Dhabi" --session "R"
```

**Options:**
- `--year`: Race year (e.g., 2023)
- `--race`: Grand Prix name (e.g., "Abu Dhabi", "Monaco")
- `--session`: Session type ("R" for race, "Q" for qualifying)
- `--cache-dir`: Optional cache directory (default: data/raw/fastf1_cache)

The script will:
1. Connect to FastF1 API
2. Download telemetry data
3. Save raw data to `data/raw/<race_name>/`

### 2. Feature Engineering

Next, generate features for the prediction model:

```bash
python scripts/engineer_features.py --race "abu_dhabi_grand_prix" --validate
```

**Options:**
- `--race`: Race name in snake_case format
- `--validate`: Run feature validation (optional)
- `--output-dir`: Output directory (default: data/processed/features)

This step:
1. Loads raw race data
2. Calculates features like:
   - Tire degradation
   - Position changes
   - Gap to competitors
   - Track evolution
3. Validates feature quality
4. Saves processed features

### 3. Making Predictions

Run the prediction model:

```bash
python scripts/predict_pit_stops.py \
    --race "abu_dhabi_grand_prix" \
    --model "random_forest" \
    --threshold 0.6
```

**Options:**
- `--race`: Race name
- `--model`: Model type ("random_forest" or "neural_network")
- `--threshold`: Prediction threshold (0.0-1.0)
- `--output-dir`: Output directory for predictions

The script will:
1. Load the trained model
2. Process race features
3. Generate pit window predictions
4. Save predictions and visualizations

### 4. Viewing Results

Launch the Streamlit dashboard:

```bash
streamlit run frontend/pit_stop_app.py
```

The dashboard shows:
1. Predicted pit windows
2. Tire degradation curves
3. Position changes
4. Strategy comparisons

## Understanding Predictions

### Prediction Output

The model generates several outputs:

```python
{
    'OptimalPitWindow': bool,      # True if optimal pit window
    'PitWindowScore': float,       # Confidence score (0.0-1.0)
    'UndercutPotential': float,    # Potential gain from undercutting
    'OvercutPotential': float,     # Potential gain from overcutting
    'RiskScore': float            # Risk assessment score
}
```

### Visualization Types

1. **Pit Window Timeline**
   - X-axis: Lap number
   - Y-axis: Pit window score
   - Highlighted regions: Optimal windows

2. **Tire Performance**
   - Degradation curves
   - Compound comparison
   - Temperature impact

3. **Strategy Comparison**
   - Position impact
   - Time delta
   - Risk assessment

## Best Practices

### 1. Data Quality

Ensure good predictions by:
- Using recent race data
- Validating feature quality
- Checking for missing values

### 2. Model Selection

Choose the appropriate model:
- Random Forest: Better for most races
- Neural Network: Better for unusual conditions

### 3. Threshold Tuning

Adjust prediction threshold based on:
- Track characteristics
- Weather conditions
- Race situation

## Troubleshooting

### Common Issues

1. **Missing Data**
   ```
   Error: No telemetry data found
   ```
   - Check internet connection
   - Verify race name spelling
   - Clear FastF1 cache

2. **Poor Predictions**
   ```
   Warning: Low confidence predictions
   ```
   - Validate feature quality
   - Adjust prediction threshold
   - Check for unusual conditions

3. **Performance Issues**
   ```
   Warning: Slow prediction time
   ```
   - Reduce feature set
   - Use faster model
   - Clear memory cache

### Getting Help

If you encounter issues:
1. Check the logs in `logs/`
2. Review validation reports
3. Create GitHub issue with:
   - Error messages
   - Race details
   - Steps to reproduce

## Advanced Usage

### 1. Custom Features

Add custom features in `src/data/feature_engineering.py`:

```python
def calculate_custom_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate custom race feature."""
    df['CustomFeature'] = # Your calculation
    return df
```

### 2. Model Tuning

Adjust model parameters in `configs/model_config.yaml`:

```yaml
random_forest:
  n_estimators: 200
  max_depth: 15
  min_samples_split: 5
```

### 3. Batch Predictions

Process multiple races:

```bash
python scripts/batch_predict.py \
    --year 2023 \
    --races "abu_dhabi,monaco,singapore" \
    --model "random_forest"
```

## Next Steps

- Read [Model Training](./model_training.md) guide
- Check [API Reference](../api/index.md)
- View [Example Notebooks](../notebooks/) 