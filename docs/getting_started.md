# Getting Started with PitGenius

This guide will help you get started with PitGenius, from installation to running your first prediction.

## Prerequisites

Before installing PitGenius, ensure you have the following:

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- FastF1 API access (for data collection)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PitGenius.git
cd PitGenius
```

2. Create and activate a virtual environment:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Unix/macOS
venv\Scripts\activate     # On Windows

# Or using conda
conda env create -f env.yml
conda activate pitgenius
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root:
```bash
cp .env.example .env
```

2. Edit the `.env` file with your configuration:
```
# API Configuration
FASTF1_CACHE_DIR=./data/raw/fastf1_cache

# Model Configuration
MODEL_VERSION=2023
RANDOM_SEED=42

# Logging Configuration
LOG_LEVEL=INFO
```

## Quick Start

### 1. Data Collection

Collect race data using the FastF1 API:

```bash
python scripts/collect_race_data.py --year 2023 --race "Abu Dhabi"
```

### 2. Feature Engineering

Generate features for the collected race:

```bash
python scripts/engineer_features.py --race "abu_dhabi_grand_prix"
```

### 3. Make Predictions

Run predictions for a race:

```bash
python scripts/predict_pit_stops.py --race "abu_dhabi_grand_prix"
```

### 4. View Results

Launch the Streamlit web interface to view predictions:

```bash
streamlit run frontend/pit_stop_app.py
```

## Project Structure

The key directories in the project are:

```
PitGenius/
├── data/               # Data storage
│   ├── raw/           # Raw race data
│   ├── processed/     # Processed features
│   └── validation/    # Validation reports
├── models/            # Trained models
├── src/               # Source code
│   ├── data/         # Data processing
│   ├── models/       # Model implementations
│   └── visualization/ # Visualization tools
├── frontend/          # Web interface
└── docs/             # Documentation
```

## Next Steps

- Read the [User Guides](./user_guides/index.md) for detailed usage instructions
- Check out the [API Reference](./api/index.md) for detailed code documentation
- Visit the [Development Guide](./development/index.md) to contribute to the project

## Troubleshooting

If you encounter any issues:

1. Check the logs in `logs/` directory
2. Verify your Python version and dependencies
3. Ensure your `.env` file is properly configured
4. Check the [FAQ](./faq.md) for common issues

For additional help, please create an issue on GitHub with:
- Your system information
- Steps to reproduce the issue
- Error messages and logs 