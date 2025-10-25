#!/usr/bin/env bash
# Exit on error
set -o errexit

# Modify this line as needed for your project
pip install -r requirements.txt

# Apply any outstanding database migrations (if you add a database later)
# python manage.py migrate
