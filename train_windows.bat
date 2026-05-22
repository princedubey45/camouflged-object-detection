@echo off
setlocal

call .venv\Scripts\activate.bat
python train.py --config config.yaml
