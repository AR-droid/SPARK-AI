#!/usr/bin/env bash
# Exit on error
set -o errexit

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install system dependencies for python-magic
apt-get update && apt-get install -y libmagic1

# Create necessary directories
mkdir -p uploads

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=production

# Apply any outstanding database migrations (if you add a database later)
# python manage.py migrate
