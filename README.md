# Real-time Whisper Audio Transcriber

A high-performance, near-realtime audio transcription tool using OpenAI Whisper with intelligent Voice Activity Detection (VAD) based chunking.

## ✨ Features

- **🎯 Near-realtime transcription** with GPU acceleration
- **🎙️ Smart audio chunking** - only cuts audio during silence to prevent word clipping
- **📱 Live display** with rich terminal interface showing transcripts and statistics
- **⚙️ Highly configurable** via YAML config files and command-line options
- **🖥️ Multi-device support** - works with VB-Audio Virtual Cable, microphones, etc.
- **📊 Performance monitoring** - GPU usage, processing times, throughput metrics
- **🔧 Professional logging** with file and console output

## 🚀 Performance

With a modern NVIDIA GPU (RTX 2070+), expect:
- **Real-time Factor < 0.1** (processes audio 10x faster than realtime)
- **Low latency** - transcription appears within 1-3 seconds of speech
- **Word boundary preservation** - VAD-based chunking prevents word clipping
- **Efficient memory usage** - optimized for continuous operation

## 📋 Prerequisites

- **Windows 10/11** (tested)
- **Python 3.12+** (tested with Python 3.12.5)
- **NVIDIA GPU** with CUDA support (optional, but highly recommended)
- **Audio input device** (microphone, VB-Audio Virtual Cable, etc.)
- **Git** (for version control, optional)

## 🛠️ Installation

### Step 1: Setup Project Directory

```powershell
# Option 1: Clone directly (recommended)
git clone https://github.com/rmacmorran/whispertranscriber.git whisper-transcriber
cd whisper-transcriber

# Option 2: Create empty directory first, then clone
# mkdir whisper-transcriber
# cd whisper-transcriber
# git clone https://github.com/rmacmorran/whispertranscriber.git .
```

### Step 2: Create Virtual Environment

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Verify activation (should show (venv) in prompt)
```

### Step 3: Install Dependencies

```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (for NVIDIA GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies from requirements.txt
pip install -r requirements.txt
```

> **Note:** PyTorch requires a special index URL for CUDA support, so it's installed separately. All other dependencies are managed through `requirements.txt` for consistency.

### Step 4: Test Installation

```powershell
# Test GPU acceleration
python test_whisper.py

# List available audio devices  
python audio_devices.py

# Test audio capture (optional)
python test_audio_buffer.py
```

### Step 5: Configure for Your System

```powershell
# List audio devices to find your device ID
python whisper-transcriber.py --list-devices

# Edit config.yaml to set your audio device
# For VB-Audio Virtual Cable, typically device_index: 31
```

## 🎮 Usage

### Basic Usage

```powershell
# Run with default settings (uses config.yaml)
python whisper-transcriber.py

# List available audio devices
python whisper-transcriber.py --list-devices

# Use specific audio device (e.g., VB-Audio Virtual Cable)
python whisper-transcriber.py --device 31

# Use different model size
python whisper-transcriber.py --model small

# Specify language (or auto-detect)
python whisper-transcriber.py --language en
```

### Configuration

The application uses `config.yaml` for configuration. Key settings:

```yaml
audio:
  device_index: 31        # VB-Audio Virtual Cable (WASAPI)
  sample_rate: 48000      # Match VB-Audio device (auto-resampled to 16kHz for Whisper)

vad:
  threshold: 0.5          # VAD sensitivity (0.1-1.0)
  min_silence_ms: 300     # Minimum silence to cut chunks
  chunk_overlap: 0.5      # Overlap between chunks (prevents word loss)

whisper:
  model_size: "base"      # tiny, base, small, medium, large
  device: "auto"          # Use GPU if available
  language: null          # Auto-detect language
```

### With VB-Audio Virtual Cable

1. **Install VB-Audio Virtual Cable** (if not already installed)
2. **Route audio** to Virtual Cable:
   - Set VB-Audio Virtual Cable as default output device, OR
   - Route specific applications to Virtual Cable
3. **Run transcriber:**
   ```powershell
   python whisper-transcriber.py --device 31  # Use WASAPI device ID
   ```

## 📊 Interface

The application provides a rich real-time interface with three sections:

### 🎯 System Status
- Running status and uptime
- Model and device information
- Current configuration

### 📝 Live Transcript  
- Real-time transcription results
- Timestamps for each transcription
- Confidence scores (optional)
- Automatically scrolls with new content

### 📈 Performance Statistics
- Audio buffer status
- Transcription throughput
- GPU memory usage and utilization
- Processing times and queue status

## 🎛️ Command Line Options

```
python whisper-transcriber.py [options]

Options:
  -h, --help           Show help message
  -c, --config FILE    Configuration file (default: config.yaml)
  -l, --list-devices   List audio input devices and exit
  -d, --device ID      Audio input device ID (overrides config)
  -m, --model SIZE     Whisper model size (tiny|base|small|medium|large)
  --language LANG      Language code (en, es, fr, etc.) or auto
```

## 🔧 Advanced Configuration

### Model Sizes and Performance

| Model | Size | VRAM | Speed | Accuracy |
|-------|------|------|-------|----------|
| tiny  | 39 MB | ~1 GB | Fastest | Basic |
| base  | 74 MB | ~1 GB | Fast | Good |
| small | 244 MB | ~2 GB | Medium | Better |
| medium| 769 MB | ~5 GB | Slow | Great |
| large | 1550 MB | ~10 GB | Slowest | Best |

**Recommended:** `base` for real-time use, `small` for better accuracy.

### VAD Settings

- **threshold**: Higher = more sensitive to speech (0.1-1.0)
- **min_silence_ms**: Minimum silence duration to end chunk (100-1000ms)
- **chunk_overlap**: Overlap between chunks to prevent word loss (0.2-1.0s)

### Audio Devices

Use `python whisper-transcriber.py --list-devices` to see available devices:
- **VB-Audio Virtual Cable** (recommended for app audio)
- **Microphones** (for live speech)
- **Realtek/USB Audio** (for hardware inputs)

## 🚨 Troubleshooting

### Common Issues

1. **No audio detected:**
   - Check device ID with `--list-devices`
   - Ensure audio is routed to the selected device
   - Test with `python test_audio_buffer.py`

2. **Slow performance:**
   - Use smaller model (`tiny` or `base`)
   - Verify GPU acceleration with `nvidia-smi`
   - Check CUDA installation

3. **Word clipping:**
   - Increase `chunk_overlap` in config
   - Adjust `min_silence_ms` for your audio

4. **Import errors:**
   - Activate virtual environment: `.\venv\Scripts\Activate.ps1`
   - Reinstall dependencies: `pip install -r requirements.txt`

### Testing Components

```powershell
# Test individual components
python test_whisper.py       # Test Whisper + GPU
python test_audio_buffer.py  # Test audio capture
python test_vad_chunker.py   # Test VAD chunking (Ctrl+C to skip live test)
```

## 📁 Project Structure

```
whisper-transcriber/
├── whisper-transcriber.py  # Main application
├── config.yaml          # Configuration file
├── whisper_engine.py    # Whisper transcription engine
├── vad_chunker.py       # VAD-based audio chunker
├── audio_buffer.py      # Audio capture buffer
├── audio_devices.py     # Device management utilities
├── test_*.py           # Test scripts
├── logs/               # Log files
└── venv/               # Python virtual environment
```

## 🎯 Performance Tuning

### For Maximum Speed
```yaml
whisper:
  model_size: "tiny"
  beam_size: 1
vad:
  min_chunk_duration: 1.5
  max_chunk_duration: 8.0
```

### For Maximum Accuracy  
```yaml
whisper:
  model_size: "small" 
  beam_size: 3
vad:
  threshold: 0.3
  min_silence_ms: 500
```

### For Balanced Performance
```yaml
whisper:
  model_size: "base"
  beam_size: 1
vad:
  threshold: 0.5
  min_silence_ms: 300
```

## 🔄 Integration Examples

### Transcribe YouTube Videos
1. Play YouTube video
2. Set VB-Audio Virtual Cable as Windows default audio output
3. Run: `python whisper-transcriber.py --device 31`

### Transcribe Zoom/Teams Calls  
1. Configure Zoom/Teams to output to VB-Audio Virtual Cable
2. Run transcriber with Virtual Cable as input
3. Transcription appears in real-time during calls

### Voice Recording
```powershell
# Use microphone directly
python whisper-transcriber.py --device 1  # Replace with your mic device ID
```

## 🔄 Batch Processing with PowerShell Script

Included with the project is `batch-transcribe.ps1`, a powerful PowerShell script that can transcribe multiple video/audio files in bulk.

### Features
- **📁 Bulk processing** - transcribes all video/audio files in a directory
- **⚙️ Flexible options** - timestamps, model size, output directory
- **📊 Progress tracking** - shows progress and timing for each file
- **🎯 Smart skipping** - avoids re-transcribing existing files (unless `-Force`)
- **🌈 Color output** - easy-to-read progress indicators
- **📂 Works anywhere** - automatically finds whisper-transcriber.py

### Basic Usage
```powershell
# Transcribe all media files in current directory
.\batch-transcribe.ps1

# Include timestamps in transcripts
.\batch-transcribe.ps1 -Timestamps

# Use larger model for better accuracy
.\batch-transcribe.ps1 -Model large -Timestamps

# Custom output directory
.\batch-transcribe.ps1 -OutputDir "my_transcripts" -Force
```

### Running from Any Directory
The batch script can run from any directory - it will automatically find your whisper-transcriber installation:

```powershell
# Option 1: Copy script to media folder
copy "C:\path\to\whisper-transcriber\batch-transcribe.ps1" .
.\batch-transcribe.ps1 -Timestamps -Model large

# Option 2: Set environment variable (recommended)
$env:WHISPER_TRANSCRIBER_PATH = "C:\path\to\whisper-transcriber\whisper-transcriber.py"
C:\any\folder\batch-transcribe.ps1 -Timestamps

# Option 3: Run with full path
C:\path\to\whisper-transcriber\batch-transcribe.ps1 -Timestamps
```

### Supported Formats
**Video:** .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm, .m4v  
**Audio:** .mp3, .wav, .flac, .aac, .ogg, .m4a, .wma

### Batch Script Options
```powershell
.\batch-transcribe.ps1 [OPTIONS]

Options:
  -OutputDir <path>   Output directory (default: current directory)
  -Timestamps         Include timestamps in output
  -Model <size>       Whisper model (tiny/base/small/medium/large)
  -Force              Overwrite existing transcripts
  -Help               Show help message
```

### Example Output
```
Using transcriber script: C:\whisper-transcriber\whisper-transcriber.py
Scanning for video and audio files...
Found 3 files to process:
   lecture1.mp4 (45.2 MB)
   interview.wav (12.8 MB)  
   presentation.m4v (89.1 MB)

Starting batch transcription...
Model: large
Timestamps: Yes
Output: transcripts\

[1/3] (0%) Processing: lecture1.mp4
   Transcribing...
   Success! Transcript saved (2.1 KB) - took 02:15

[2/3] (33.3%) Processing: interview.wav
   Transcribing...
   Success! Transcript saved (892 bytes) - took 00:45
```

## 📄 License

This project uses open-source components:
- **OpenAI Whisper** (MIT License)
- **faster-whisper** (MIT License) 
- **Silero VAD** (MIT License)

---

**🎉 Enjoy real-time transcription with GPU acceleration!**
