# simple_cnn_train.ps1
# PowerShell script to run the simplified CNN model training

# Navigate to the project root directory
Set-Location $PSScriptRoot

# Set TensorFlow memory growth option
$env:TF_FORCE_GPU_ALLOW_GROWTH = "true"

# Create virtual environment if it doesn't exist
if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
    
    # Activate virtual environment
    & .venv\Scripts\Activate.ps1
    
    # Install required packages
    Write-Host "Installing required packages..." -ForegroundColor Yellow
    pip install tensorflow numpy scikit-learn matplotlib
} else {
    # Activate virtual environment
    & .venv\Scripts\Activate.ps1
}

# Run the training script
Write-Host "Starting simplified CNN model training..." -ForegroundColor Cyan
python simple_cnn_train.py

# Check if training was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "Training completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
}

# Deactivate virtual environment
deactivate
