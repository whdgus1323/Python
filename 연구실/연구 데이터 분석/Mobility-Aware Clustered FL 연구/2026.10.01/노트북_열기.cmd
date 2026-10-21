@echo off
cd /d "%~dp0"
uv run --no-project --python 3.13 --with numpy==2.4.6 --with duckdb==1.5.6 --with pyarrow==25.0.1 --with matplotlib==3.11.2 --with ipykernel==7.4.0 --with jupyterlab==4.6.4 jupyter lab "FL_Log_Graphs.ipynb"
pause
