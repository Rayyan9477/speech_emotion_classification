# train_cnn_model.ps1
# PowerShell script to run the CNN model training

# Navigate to the project root directory
Set-Location $PSScriptRoot

# Clear TensorFlow memory (if needed)
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"

# Create virtual environment if it doesn't exist
if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv .venv
    
    # Activate virtual environment
    & .venv\Scripts\Activate.ps1
    
    # Install required packages
    Write-Host "Installing required packages..."
    pip install -r requirements.txt
} else {
    # Activate virtual environment
    & .venv\Scripts\Activate.ps1
}

# Run the training script
Write-Host "Starting CNN model training..."
python train_cnn_model.py

# Check if training was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
}

# Deactivate virtual environment
deactivate
