#!/bin/bash

# Exit on error
set -e

echo "Setting up PitGenius development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories if they don't exist
echo "Creating project directories..."
mkdir -p data/{raw,processed,interim,external}
mkdir -p models/{trained,evaluations}
mkdir -p logs

# Set up pre-commit hooks if .git directory exists
if [ -d ".git" ]; then
    echo "Setting up pre-commit hooks..."
    pip install pre-commit
    pre-commit install
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

echo "Setup complete! Activate the virtual environment with:"
echo "source venv/bin/activate"
