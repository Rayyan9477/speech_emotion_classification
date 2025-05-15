# fix_imports.ps1
# PowerShell script to fix imports in all model files

# Navigate to the project root directory
Set-Location $PSScriptRoot

# Define a function to fix imports in a Python file
function Fix-Imports {
    param (
        [string]$FilePath
    )
    
    Write-Host "Fixing imports in $FilePath"
    
    # Read the file content
    $content = Get-Content -Path $FilePath -Raw
    
    # Fix common import issues
    
    # Fix tensorflow imports
    $content = $content -replace "from tensorflow\.keras", "from keras"
    $content = $content -replace "import tensorflow as tf", "import tensorflow as tf`nimport numpy as np"
    
    # Fix relative imports
    $content = $content -replace "from models import", "from .emotion_model import"
    $content = $content -replace "from trainer import", "from .trainer import"
    
    # Fix monkey patch imports
    $content = $content -replace "import backup\.tf_patch", "try:`n    from src.utils.monkey_patch import monkeypatch`n    monkeypatch()`nexcept ImportError:`n    pass"
    
    # Write the file content back
    $content | Set-Content -Path $FilePath
    
    Write-Host "Fixed imports in $FilePath" -ForegroundColor Green
}

# List of files to fix
$files = @(
    "src/models/emotion_model.py",
    "src/models/trainer.py",
    "src/models/model_manager.py",
    "src/models/optimizer.py"
)

# Fix imports in each file
foreach ($file in $files) {
    Fix-Imports -FilePath $file
}

Write-Host "All imports fixed!" -ForegroundColor Green
