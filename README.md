# F1 Strategy Optimizer

A comprehensive system for Formula 1 race strategy optimization using machine learning techniques.

## Overview

This project aims to provide pre-race strategy recommendations for Formula 1 teams, along with the ability to compare predicted strategies to actual race outcomes. The system leverages machine learning to optimize race strategies based on various factors while accounting for driver/team preferences and enabling scenario simulations.

## Key Features

- Pre-Race Strategy Recommendation
- Driver/Team Preference Modeling
- Scenario Simulation and Comparison
- Performance Metrics and Reporting

## Technical Stack

- **Data Collection**: FastF1 Python library, Ergast API
- **Machine Learning**: Scikit-learn, TensorFlow/PyTorch, XGBoost
- **Frontend**: Streamlit
- **Visualization**: Plotly/Dash, Matplotlib
- **Core Technologies**: Python, Pandas, NumPy

## Project Structure

```
f1_strategy_optimizer/
├── src/               # Source code
├── app/               # Web application
├── data/              # Data collection and processing
├── tests/             # Unit and integration tests
├── docs/              # Documentation
└── config/            # Configuration files
```

## Setup

1. Create a conda environment:
   ```bash
   conda env create -f env.yml
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app/main.py
   ```

## Documentation

- User Guide: See `docs/user_guide.md`
- Technical Documentation: See `docs/technical_documentation.md`

## License

This project is licensed under the MIT License.
