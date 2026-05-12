# install.ps1 — Anvil Audio self-contained installer for Windows
# Run with: powershell -ExecutionPolicy Bypass -File install.ps1
param(
    [switch]$SkipAceStep
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$VenvDir = ".venv"
$PythonExe = "python"

Write-Host "=== Anvil Audio Installer ===" -ForegroundColor Cyan
Write-Host "Platform: Windows x64"
Write-Host ""

# ── Python version check ──────────────────────────────────────────────────────
$PyVer = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python not found. Install Python 3.12 or 3.13 from https://python.org"
    exit 1
}
$Parts = $PyVer.Split(".")
$PyMajor = [int]$Parts[0]
$PyMinor = [int]$Parts[1]
if ($PyMajor -lt 3 -or ($PyMajor -eq 3 -and $PyMinor -lt 12) -or ($PyMajor -eq 3 -and $PyMinor -gt 13)) {
    Write-Error "Python 3.12 or 3.13 required (found $PyVer). Python 3.14+ is too new for several ML dependencies. Download 3.13 from https://python.org"
    exit 1
}
Write-Host "Python $PyVer — OK"

# ── Virtual environment ───────────────────────────────────────────────────────
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment at $VenvDir..."
    & $PythonExe -m venv $VenvDir
}
$Pip = "$VenvDir\Scripts\pip.exe"
Write-Host "Using venv: $VenvDir"
& $Pip install --upgrade pip --quiet

# ── PyTorch (CUDA) ───────────────────────────────────────────────────────────
Write-Host "Installing PyTorch for Windows CUDA..."
& $Pip install torch torchaudio `
    --index-url https://download.pytorch.org/whl/cu128 --quiet

# ── Anvil Audio ───────────────────────────────────────────────────────────────
Write-Host "Installing anvil-audio..."
& $Pip install -e . --quiet

# ── ACE-Step (optional) ───────────────────────────────────────────────────────
if (-not $SkipAceStep) {
    $Answer = Read-Host "Install ACE-Step for music generation? [y/N]"
    if ($Answer -match "^[Yy]$") {
        Write-Host "Installing nano-vllm (required on Windows)..."
        & $Pip install `
            "git+https://github.com/PiMPStudios/ACE-Step-1.5.git#subdirectory=acestep/third_parts/nano-vllm"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "nano-vllm install failed — ACE-Step may not work correctly."
        }

        Write-Host "Installing ACE-Step..."
        & $Pip install `
            "ace-step @ git+https://github.com/PiMPStudios/ACE-Step-1.5.git"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "ACE-Step installed." -ForegroundColor Green
        } else {
            Write-Warning "ACE-Step install failed. Check errors above."
        }
    }
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Installation complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Activate your environment:"
Write-Host "  $VenvDir\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Check your setup:"
Write-Host "  anvil setup"
Write-Host ""
Write-Host "Generate audio:"
Write-Host "  anvil generate --model stable-audio-open-1.0 --prompt ""rain on a tin roof"""
