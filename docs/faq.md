# Frequently Asked Questions (FAQ)

## General Questions

### What is PitGenius?
PitGenius is a machine learning-based system that predicts optimal pit stop windows during Formula 1 races. It analyzes various factors like tire degradation, track position, and weather conditions to recommend the best timing for pit stops.

### How accurate are the predictions?
The model achieves approximately 85-90% accuracy on test data. However, accuracy can vary depending on race conditions, track characteristics, and data quality. We recommend using predictions as guidance rather than absolute truth.

### Which races are supported?
PitGenius supports all Formula 1 races from 2018 onwards. However, prediction quality is best for recent races (2021 onwards) due to changes in car regulations and tire compounds.

## Installation & Setup

### What are the system requirements?
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space
- Internet connection for FastF1 API

### Why am I getting package conflicts?
Common causes:
1. Python version mismatch
2. Outdated pip
3. Conflicting dependencies

Solution:
```bash
# Create fresh environment
python -m venv venv
source venv/bin/activate

# Update pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

### How do I fix FastF1 API issues?
1. Clear the cache:
   ```bash
   rm -rf data/raw/fastf1_cache/*
   ```
2. Check internet connection
3. Verify API endpoint status
4. Update FastF1 package

## Data Collection

### Why is data download slow?
FastF1 API speed depends on:
- Internet connection
- Server load
- Data size
- Cache status

Tips:
- Use cache directory
- Download during off-peak hours
- Check network speed

### Missing telemetry data?
Common causes:
1. Incorrect race name
2. Session not available
3. API issues

Solution:
```bash
# Verify race name
python scripts/collect_race_data.py --list-races

# Try different session
python scripts/collect_race_data.py --race "Abu Dhabi" --session "Q"

# Clear cache and retry
rm -rf data/raw/fastf1_cache/*
```

## Feature Engineering

### Why are features invalid?
Common issues:
1. Missing raw data
2. Data quality problems
3. Calculation errors

Check validation report:
```bash
python scripts/engineer_features.py --race "abu_dhabi_grand_prix" --validate
```

### How do I add custom features?
1. Create feature function:
   ```python
   # src/data/feature_engineering.py
   
   def calculate_custom_feature(df: pd.DataFrame) -> pd.DataFrame:
       df['CustomFeature'] = # Your calculation
       return df
   ```

2. Add to pipeline:
   ```python
   features = (
       df.pipe(calculate_tire_features)
         .pipe(calculate_position_features)
         .pipe(calculate_custom_feature)
   )
   ```

## Model Training

### Why is model performance poor?
Common causes:
1. Class imbalance
2. Feature quality
3. Hyperparameter tuning

Solutions:
- Use class weights
- Validate features
- Tune hyperparameters:
  ```python
  model = RandomForestModel({
      'n_estimators': 200,
      'max_depth': 15,
      'class_weight': 'balanced'
  })
  ```

### How do I improve predictions?
1. Feature engineering:
   - Add domain knowledge
   - Create interaction features
   - Handle outliers

2. Model tuning:
   - Cross-validation
   - Grid search
   - Ensemble methods

3. Data quality:
   - Clean data
   - Handle missing values
   - Remove noise

## Visualization

### Dashboard not loading?
Check:
1. Streamlit installation
2. Port availability
3. Data paths

Solution:
```bash
# Reinstall Streamlit
pip install --upgrade streamlit

# Check port
lsof -i :8501

# Verify data
ls data/processed/features/
```

### How do I customize plots?
1. Modify plot settings:
   ```python
   # src/visualization/pit_stop_viz.py
   
   PLOT_CONFIG = {
       'figure.figsize': (12, 8),
       'axes.titlesize': 14,
       'style': 'dark'
   }
   ```

2. Add custom plots:
   ```python
   def create_custom_plot(data: pd.DataFrame) -> plt.Figure:
       fig, ax = plt.subplots()
       # Your plotting code
       return fig
   ```

## Deployment

### How do I deploy to production?
Options:
1. Docker container:
   ```bash
   docker build -t pitgenius .
   docker run -p 8501:8501 pitgenius
   ```

2. Cloud service (AWS/GCP):
   - Use provided deployment scripts
   - Set up CI/CD pipeline
   - Configure monitoring

### Memory issues in production?
1. Optimize data loading:
   ```python
   # Use chunks
   for chunk in pd.read_csv('data.csv', chunksize=1000):
       process_chunk(chunk)
   ```

2. Clear cache:
   ```python
   import gc
   gc.collect()
   ```

3. Monitor memory:
   ```python
   import psutil
   print(psutil.Process().memory_info().rss / 1024 / 1024)
   ```

## Contributing

### How do I contribute?
1. Fork repository
2. Create feature branch
3. Make changes
4. Submit pull request

Guidelines:
- Follow style guide
- Add tests
- Update docs
- Write clear commit messages

### Where do I report bugs?
1. Check existing issues
2. Create new issue with:
   - Error message
   - Steps to reproduce
   - System info
   - Logs

## Support

### Where can I get help?
1. Documentation:
   - User guides
   - API reference
   - Examples

2. Community:
   - GitHub issues
   - Discussion forum
   - Stack Overflow

3. Contact:
   - Email support
   - Bug reports
   - Feature requests 