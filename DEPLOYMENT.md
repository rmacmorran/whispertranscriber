# 🚀 Whisper Transcriber Deployment Guide

This guide covers how to deploy the Whisper Transcriber to another machine.

## 📦 Files Required for Deployment

### **Essential Files (Must Deploy):**
```
whisper-transcriber/
├── whisper-transcriber.py   # Main application (renamed from main.py)
├── whisper_engine.py        # Whisper transcription engine
├── vad_chunker.py           # VAD-based audio chunker
├── audio_buffer.py          # Audio capture buffer
├── audio_devices.py         # Device management utilities
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies
└── README.md                # Documentation
```

### **Optional Files (Recommended):**
```
├── test_whisper.py          # Test Whisper functionality
├── test_audio_buffer.py     # Test audio capture
├── test_vad_chunker.py      # Test VAD chunking
└── DEPLOYMENT.md            # This deployment guide
```

### **Files NOT to Deploy:**
```
├── venv/                    # Virtual environment (recreate on target)
├── logs/                    # Log files (will be created)
├── .gitignore              # Git-specific
└── __pycache__/            # Python cache (will be recreated)
```

## 🎯 Deployment Methods

### Method 1: Copy Project Folder
```bash
# Copy entire project folder (excluding venv)
xcopy /E /I "C:\Users\rmacmorran\projects\whisper-transcriber" "\\target-machine\path\whisper-transcriber"
# OR use robocopy
robocopy "C:\Users\rmacmorran\projects\whisper-transcriber" "\\target-machine\path\whisper-transcriber" /E /XD venv __pycache__ logs
```

### Method 2: ZIP Archive
1. Create ZIP of project folder (exclude `venv/`, `logs/`, `__pycache__/`)
2. Transfer ZIP to target machine
3. Extract to desired location

### Method 3: Git Repository (Recommended)
```bash
# Initialize git repo (if not already done)
git init
git add .
git commit -m "Initial commit"

# Deploy via git clone on target machine
git clone <repository-url>
```

## 🛠️ Installation on Target Machine

### Step 1: Prerequisites
Target machine needs:
- **Windows 10/11**
- **Python 3.12+** 
- **NVIDIA GPU with CUDA** (optional but recommended)
- **Audio input device** (microphone, VB-Audio Virtual Cable, etc.)

### Step 2: Setup Project
```powershell
# Navigate to project directory
cd path\to\whisper-transcriber

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Step 4: Test Installation
```powershell
# Test GPU acceleration
python test_whisper.py

# List available audio devices  
python audio_devices.py

# Test the application
python whisper-transcriber.py --list-devices
```

### Step 5: Configure for Target System
```powershell
# Find audio device ID
python whisper-transcriber.py --list-devices

# Edit config.yaml to set correct device_index
# OR use command line override:
python whisper-transcriber.py --device <device_id>
```

## ⚙️ Configuration for Different Environments

### For Different Audio Devices
Update `config.yaml`:
```yaml
audio:
  device_index: <device_id>  # From --list-devices
  sample_rate: 48000         # Match device sample rate
```

### For Different GPU/CPU Setup
```yaml
whisper:
  device: "auto"        # Auto-detect GPU/CPU
  # OR specify:
  device: "cuda"        # Force GPU
  device: "cpu"         # Force CPU
  compute_type: "float16"  # GPU: float16, CPU: float32
```

### For Different Performance Requirements
```yaml
whisper:
  model_size: "base"    # tiny/base/small/medium/large
  beam_size: 1          # 1-5, higher = better quality, slower
```

## 🚀 Running the Application

### Basic Usage
```powershell
# GUI mode (default)
python whisper-transcriber.py

# Console mode
python whisper-transcriber.py -q

# File transcription
python whisper-transcriber.py -i "audio.mp4" -o "transcript.txt" -tq
```

### With Custom Configuration
```powershell
# Use specific device
python whisper-transcriber.py -d 31

# Use different model
python whisper-transcriber.py -m small

# Combine options
python whisper-transcriber.py -d 31 -m small -q
```

## 🎯 Platform-Specific Notes

### Windows
- Ensure Windows Audio is working
- VB-Audio Virtual Cable requires installation on target machine
- CUDA drivers must be installed for GPU acceleration

### Audio Device Setup
1. **For Microphone Input:**
   - Use built-in microphone device ID
   - Typically device ID 1 or 2

2. **For Application Audio (VB-Audio Virtual Cable):**
   - Install VB-Audio Virtual Cable on target machine
   - Device ID typically 31 (WASAPI)
   - Configure applications to output to Virtual Cable

3. **For System Audio:**
   - Use "Stereo Mix" if available
   - May require enabling in Windows Sound Settings

## 🔧 Troubleshooting Deployment

### Common Issues

#### 1. **Import Errors**
```
ModuleNotFoundError: No module named 'faster_whisper'
```
**Solution:** Activate virtual environment and reinstall requirements
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### 2. **CUDA/GPU Issues**
```
RuntimeError: No CUDA devices available
```
**Solution:** Install CUDA drivers or use CPU mode
```yaml
whisper:
  device: "cpu"
  compute_type: "float32"
```

#### 3. **Audio Device Not Found**
```
Device 31 not found
```
**Solution:** List available devices and update configuration
```powershell
python whisper-transcriber.py --list-devices
# Update config.yaml with correct device_index
```

#### 4. **Permission Errors**
```
PermissionError: [WinError 5] Access is denied
```
**Solution:** Run as administrator or check file permissions

### Testing Individual Components
```powershell
# Test Whisper engine
python test_whisper.py

# Test audio capture  
python test_audio_buffer.py

# Test VAD chunking
python test_vad_chunker.py
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] Test application on source machine
- [ ] Verify all essential files are present
- [ ] Document current configuration settings
- [ ] Create backup of working configuration

### During Deployment
- [ ] Copy all essential files to target machine
- [ ] Create virtual environment on target machine
- [ ] Install dependencies via requirements.txt
- [ ] Install CUDA drivers if using GPU

### Post-Deployment
- [ ] Test basic functionality (`python whisper-transcriber.py --help`)
- [ ] List and configure audio devices
- [ ] Test transcription with sample audio
- [ ] Verify GPU acceleration (if applicable)
- [ ] Document target machine configuration

## 🎉 Ready to Deploy!

Your Whisper Transcriber is now ready for deployment. The key points are:

1. **Deploy the entire project** (not just main file)
2. **Recreate virtual environment** on target machine
3. **Install dependencies** via requirements.txt
4. **Configure audio devices** for target system
5. **Test thoroughly** before production use

---

**Happy transcribing! 🎤➡️📝**
