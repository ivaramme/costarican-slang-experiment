#!/bin/bash

# Deactivate existing virtualenv if active
if [ ! -z "$VIRTUAL_ENV" ]; then
  echo "Deactivating existing virtual environment..."
  deactivate
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

pip install -r requirements.txt
pip install --upgrade pip
