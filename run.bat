@echo off
powershell -ExecutionPolicy Bypass -Command "Expand-Archive -Path batch-script.zip -DestinationPath ."
python.exe batchactivity_json2csv.py