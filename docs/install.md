# Installation

Detailed install and verification notes for Anvil Audio.

## Requirements

- **Python 3.13** — recommended (3.12 also supported)
- Python 3.14+ is too new for several ML dependencies and will cause build failures
- PyTorch 2.0 or later

---

## Install

> **Python 3.13 is recommended.** Check with `python3 --version`.
> Install via `brew install python@3.13` (macOS) or from [python.org](https://python.org) (Windows).

### macOS / Linux

```bash
git clone https://github.com/PiMPStudios/anvil-audio-v2.git
cd anvil-audio-v2
python3.13 -m venv .venv
source .venv/bin/activate
bash install.sh
```

The script detects your platform, installs the right PyTorch build, adds
`pytest` for local verification, installs `yt-dlp` for dataset building,
enables MLX acceleration, local prompt intelligence, and the lightweight MLX
Whisper runtime on Apple Silicon, and optionally installs ACE-Step for music
generation — all in one step. Whisper model weights are still downloaded lazily
the first time transcription is used. No separate repo clones required.

### Windows

```powershell
git clone https://github.com/PiMPStudios/anvil-audio-v2.git
cd anvil-audio-v2
python -m venv .venv
.venv\Scripts\Activate.ps1
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Manual install (advanced)

```bash
python3.13 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

pip install .                      # core install
# Apple Silicon only: MLX acceleration + local intelligence
pip install mlx-audiogen mlx-lm
pip install yt-dlp                 # optional: YouTube dataset builder
pip install lightning-whisper-mlx  # Apple Silicon vocal transcription runtime
# or: pip install openai-whisper   # optional: cross-platform local transcription
pip install 'anvil-audio[acestep]' # optional: ACE-Step music generation
pip install pytest                 # optional: run the local test suite
```

### Verify your setup

```bash
anvil setup
.venv/bin/python -m pytest
```

`anvil setup` prints your platform, which optional packages are active, and what
models are registered. The `pytest` command runs the local smoke tests.
