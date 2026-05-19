# Voice Assistant with Wake Word Detection

A modular voice assistant system for Windows 11 with Traditional Chinese ASR support. Designed for easy migration to other platforms (e.g., Google Glass).

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VoiceAssistant                           │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ WakeWordDetector│    │  Command Handler │                │
│  └────────┬────────┘    └────────┬────────┘                │
│           │                      │                          │
│  ┌────────▼──────────────────────▼────────┐                │
│  │              ASR Interface              │                │
│  │         (asr_interface.py)              │                │
│  └────────────────┬───────────────────────┘                │
│                   │                                         │
│  ┌────────────────▼───────────────────────┐                │
│  │           WhisperASR                    │ ◄── Swappable │
│  │       (faster-whisper)                  │                │
│  └────────────────────────────────────────┘                │
│                                                             │
│  ┌────────────────────────────────────────┐                │
│  │            Audio Interface              │                │
│  │        (audio_interface.py)             │                │
│  └────────────────┬───────────────────────┘                │
│                   │                                         │
│  ┌────────────────▼───────────────────────┐                │
│  │        WindowsMicrophone                │ ◄── Swappable │
│  │          (PyAudio)                      │    (→ Glass)  │
│  └────────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration |
| `audio_interface.py` | Abstract audio interface |
| `audio_windows.py` | Windows microphone implementation |
| `asr_interface.py` | Abstract ASR interface |
| `asr_whisper.py` | Faster-Whisper implementation |
| `wake_word_detector.py` | Wake word detection |
| `voice_assistant.py` | Main orchestrator |
| `main.py` | Entry point |

## Installation

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. For GPU acceleration (recommended)

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. PyAudio on Windows

If PyAudio fails to install:
```bash
pip install pipwin
pipwin install pyaudio
```

Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

## Usage

### Basic Usage

```bash
python main.py
```

### List Audio Devices

```bash
python main.py --list-devices
```

### Use Specific Audio Device

```bash
python main.py --audio-device 2
```

### Change Model Size

```bash
# Faster but less accurate
python main.py --model small

# More accurate but slower
python main.py --model large-v3
```

### CPU Only Mode

```bash
python main.py --device cpu
```

### Custom Wake Phrase

```bash
python main.py --wake-phrase "hey assistant"
```

### Single Command Mode

```bash
python main.py --single
```

## Model Selection Guide

| Model | Speed | Accuracy | VRAM | Use Case |
|-------|-------|----------|------|----------|
| `tiny` | Fastest | Lower | ~1GB | Quick tests |
| `base` | Fast | Moderate | ~1GB | Simple commands |
| `small` | Medium | Good | ~2GB | Balanced |
| `medium` | Slower | Better | ~5GB | **Recommended for Chinese** |
| `large-v3` | Slowest | Best | ~10GB | Maximum accuracy |

For Traditional Chinese, `medium` or `large-v3` is recommended.

## Extending for Google Glass

To add Google Glass support, create `audio_glass.py`:

```python
from audio_interface import AudioInterface

class GlassMicrophone(AudioInterface):
    \"\"\"Google Glass microphone implementation.\"\"\"
    
    def initialize(self) -> bool:
        # Initialize Glass audio hardware
        pass
    
    def start_stream(self) -> bool:
        # Start Glass audio stream
        pass
    
    # ... implement other methods
```

Then update `audio_windows.py` factory:

```python
def create_audio_source(backend: str = "windows", **kwargs):
    if backend == "windows":
        return WindowsMicrophone(**kwargs)
    elif backend == "glass":
        from audio_glass import GlassMicrophone
        return GlassMicrophone(**kwargs)
```

## Configuration

Edit `config.py` to tune:

- **Audio settings**: Sample rate, chunk size
- **Wake word**: Phrase, confidence threshold
- **ASR**: Model size, language, VAD parameters
- **Recording**: Silence detection, max duration

## Performance Tips

1. **Use GPU**: 4-10x faster than CPU
2. **Enable VAD**: Filters silence, reduces processing
3. **Use lightweight wake detector**: Energy-based pre-filtering saves CPU
4. **Tune silence threshold**: Higher = faster cutoff, may clip speech
5. **Model size trade-off**: `small` for speed, `medium` for accuracy

## Troubleshooting

### No audio detected
- Check microphone permissions in Windows Settings
- Run `--list-devices` to verify input device
- Try specifying device with `--audio-device N`

### Slow transcription
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use smaller model: `--model small`
- Check GPU memory usage

### Wake word not detected
- Speak clearly after startup
- Check `confidence_threshold` in config
- Try alternative phrases (e.g., "ok glass")

## License

MIT
