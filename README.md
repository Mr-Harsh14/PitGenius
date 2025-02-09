# PitGenius: F1 Race Strategy Optimizer

A comprehensive system for providing pre-race strategy recommendations for Formula 1 teams, leveraging machine learning techniques to optimize race strategies and compare predicted strategies with actual race outcomes.

## Features

- **Pre-Race Strategy Recommendation**
  - Input pre-race conditions (qualifying positions, tire allocations, weather forecast)
  - ML-powered strategy optimization
  - Detailed pit stop and tire choice recommendations

- **Performance Metrics and Reporting**
  - Strategy quality evaluation
  - Detailed performance reports
  - Interactive visualizations

## Project Structure

```
PitGenius/
├── data/                     # Data files
├── models/                   # Saved model files
├── notebooks/               # Jupyter notebooks
├── src/                    # Source code
├── tests/                  # Test files
├── configs/                # Configuration files
├── docs/                   # Documentation
└── scripts/                # Utility scripts
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/PitGenius.git
cd PitGenius
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

2. Run the main application:
```bash
python main.py
```

## Development

- Data processing scripts are in `src/data/`
- Model development code is in `src/models/`
- Visualization tools are in `src/visualization/`

## Testing

Run tests using:
```bash
python -m pytest tests/
```

## Documentation

- API Reference: `docs/api_reference.md`
- Data Dictionary: `docs/data_dictionary.md`
- Model Architecture: `docs/model_architecture.md`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors

- Your Name (@yourusername)

## Acknowledgments

- FastF1 library
- Formula 1 for providing data access
