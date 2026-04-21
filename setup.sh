#!/usr/bin/env bash

set -e

echo "Setting up local virtual environment..."

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing requirements..."
python -m pip install -r requirements.txt

echo "Installing Streamlit for local development..."
python -m pip install streamlit==1.50.0

echo "Registering src/ on the virtualenv Python path..."
python scripts/install_src_path.py

echo ""
echo "Running database setup check..."
bash src/data_utils/data_setup.sh

echo "Setup complete."
