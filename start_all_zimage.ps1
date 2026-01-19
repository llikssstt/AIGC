# Function to start a process in a new window
function Start-ServiceWindow {
    param (
        [string]$Title,
        [string]$Command,
        [string]$Arguments,
        [string]$WorkDir = "."
    )
    
    Write-Host "Starting $Title..." -ForegroundColor Cyan
    
    # Check if we are in a conda environment
    if ($env:CONDA_PREFIX) {
        # Construct a command that runs inside the current conda environment
        # We wrap the command in 'cmd /c' to keep the window open if it crashes immediately, 
        # or use 'python' directly if we are sure it's in path.
        
        # Using Start-Process to open a new window
        Start-Process -FilePath "cmd.exe" `
            -ArgumentList "/k", "title $Title && cd /d $WorkDir && $Command $Arguments" `
            -WorkingDirectory $WorkDir `
            -WindowStyle Normal
    } else {
        Write-Warning "Conda environment not detected. Attempting to run with system path..."
        Start-Process -FilePath "cmd.exe" `
            -ArgumentList "/k", "title $Title && cd /d $WorkDir && $Command $Arguments" `
            -WorkingDirectory $WorkDir `
            -WindowStyle Normal
    }
}

$Root = Get-Location

# 1. Start LLM Server
# Note: Adjust the model path if yours is different!
$ModelPath = "E:\AIGC\models\Qwen3-1.7B" 
Start-ServiceWindow -Title "LLM Server (Port 8001)" `
    -Command "python" `
    -Arguments "-m sdxl_app.engine.simple_llm_server --model $ModelPath --port 8001" `
    -WorkDir $Root

# Wait a bit for LLM to initialize (optional, but nice)
Start-Sleep -Seconds 2

# 2. Start Backend API
Start-ServiceWindow -Title "SDXL Backend (Port 8000)" `
    -Command "python" `
    -Arguments "server_zimage.py --zimage-model models/Z-Image-Turbo-FP8 --dtype fp8" `
    -WorkDir $Root

# 3. Start Frontend
Start-ServiceWindow -Title "Frontend UI (Port 5173)" `
    -Command "npm" `
    -Arguments "run dev" `
    -WorkDir "$Root\frontend"

Write-Host "All services launch commands issued!" -ForegroundColor Green
Write-Host "Please check the opened windows for logs."
