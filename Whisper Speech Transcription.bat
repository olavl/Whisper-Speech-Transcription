@echo off
SET "BATCH_DIR=%~dp0"
cd %BATCH_DIR%
call .\Scripts\activate
start /min python app.py

