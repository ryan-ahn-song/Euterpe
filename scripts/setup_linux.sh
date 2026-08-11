#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[live]'
python -m unittest discover -s tests -v

echo "Installed. Run: hearing-assist devices"
