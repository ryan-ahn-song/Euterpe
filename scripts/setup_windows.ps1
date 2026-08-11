$ErrorActionPreference = "Stop"

py -3 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[live]"
python -m unittest discover -s tests -v

Write-Host "Installed. Run: hearing-assist devices"
