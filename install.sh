#!/usr/bin/env bash
# install.sh — Anvil Audio self-contained installer for macOS and Linux
set -euo pipefail

PYTHON=${PYTHON:-python3}
VENV_DIR=".venv"

# ── Platform detection ────────────────────────────────────────────────────────
OS=$(uname -s)
ARCH=$(uname -m)

echo "=== Anvil Audio Installer ==="
echo "Platform: $OS / $ARCH"
echo ""

# ── Python version check ──────────────────────────────────────────────────────
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 12 ) || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -gt 13 ) ]]; then
    echo "ERROR: Python 3.12 or 3.13 required (found ${PY_MAJOR}.${PY_MINOR})."
    echo "Python 3.14+ is too new for several ML dependencies."
    echo "Install via: brew install python@3.13  (macOS)"
    echo "             sudo apt install python3.13  (Linux)"
    exit 1
fi
echo "Python ${PY_MAJOR}.${PY_MINOR} — OK"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    $PYTHON -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
echo "Using venv: $VENV_DIR"

$PIP install --upgrade pip --quiet

# ── PyTorch (platform-specific) ───────────────────────────────────────────────
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "Installing PyTorch for macOS Apple Silicon..."
    $PIP install torch torchaudio --quiet

elif [[ "$OS" == "Linux" && "$ARCH" == "x86_64" ]]; then
    echo "Installing PyTorch for Linux CUDA..."
    $PIP install torch torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 --quiet

else
    echo "Installing PyTorch (CPU fallback)..."
    $PIP install torch torchaudio --quiet
fi

# ── Anvil Audio ───────────────────────────────────────────────────────────────
echo "Installing anvil-audio..."
$PIP install -e . --quiet

# ── Test tooling ──────────────────────────────────────────────────────────────
echo "Installing test tooling (pytest)..."
$PIP install pytest --quiet

# ── Dataset tooling ───────────────────────────────────────────────────────────
echo "Installing dataset tooling (yt-dlp)..."
$PIP install yt-dlp --quiet

# ── MLX acceleration (Apple Silicon only) ────────────────────────────────────
if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
    echo "Installing MLX acceleration and local intelligence tooling..."
    $PIP install mlx-audiogen mlx-lm --quiet
fi

# ── ACE-Step (optional music generation) ─────────────────────────────────────
read -r -p "Install ACE-Step for music generation? [y/N] " INSTALL_AS
if [[ "$INSTALL_AS" =~ ^[Yy]$ ]]; then

    if [[ "$OS" == "Linux" ]]; then
        echo "Installing nano-vllm (required on Linux)..."
        $PIP install \
            "git+https://github.com/ace-step/ACE-Step-1.5.git#subdirectory=acestep/third_parts/nano-vllm" \
            --ignore-requires-python --quiet
        if [[ $? -ne 0 ]]; then
            echo "WARNING: nano-vllm install failed — ACE-Step may not work correctly."
        fi
    fi

    echo "Installing ACE-Step..."
    $PIP install \
        "ace-step @ git+https://github.com/ace-step/ACE-Step-1.5.git" \
        --ignore-requires-python --quiet
    if [[ $? -eq 0 ]]; then
        echo "ACE-Step installed."
    else
        echo "WARNING: ACE-Step install failed. Check errors above."
    fi
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installation complete ==="
echo ""
echo "Activate your environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Check your setup:"
echo "  anvil setup"
echo ""
echo "Generate audio:"
echo "  anvil generate --model stable-audio-open-1.0-mlx --prompt \"rain on a tin roof\""
