#!/bin/bash

echo "Creating python virtual environment..."
python -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing required packages from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete! To activate this environment in the future, run: source .venv/bin/activate"
