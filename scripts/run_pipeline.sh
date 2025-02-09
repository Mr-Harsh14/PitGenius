#!/bin/bash

# Exit on error
set -e

# Load environment variables
source .env

echo "Starting PitGenius data pipeline..."

# Activate virtual environment
source venv/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Start logging
exec 1> >(tee -a "logs/pipeline_$(date +%Y%m%d_%H%M%S).log") 2>&1

echo "$(date): Pipeline started"

# Step 1: Data Collection
echo "Step 1: Collecting F1 race data..."
python -m src.data.data_collector

# Step 2: Data Cleaning
echo "Step 2: Cleaning collected data..."
python -m src.data.data_cleaner

# Step 3: Feature Engineering
echo "Step 3: Engineering features..."
python -m src.data.feature_engineer

# Step 4: Model Training
echo "Step 4: Training models..."
python -m src.models.train_model

# Step 5: Model Evaluation
echo "Step 5: Evaluating models..."
python -m src.models.evaluate_model

echo "$(date): Pipeline completed successfully"

# Deactivate virtual environment
deactivate
