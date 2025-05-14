# Add the project root to PYTHONPATH
$Env:PYTHONPATH = "$PSScriptRoot;$Env:PYTHONPATH"

# Run the trainer script
python src/models/trainer.py
