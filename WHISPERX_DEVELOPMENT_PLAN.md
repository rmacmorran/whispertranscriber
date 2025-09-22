# WhisperX Engine Integration - Development Plan

## Overview

This document outlines the complete development plan to add WhisperX engine support to the whisper-transcriber project, allowing users to choose between faster-whisper and WhisperX engines via command-line switches.

## Goals

- ✅ **Backward Compatibility**: Existing functionality remains unchanged (faster-whisper as default)
- ✅ **Optional Dependency**: Program runs without WhisperX installed when using faster-whisper
- ✅ **Feature Parity**: All existing command-line switches work with both engines where applicable
- ✅ **New Capabilities**: Speaker diarization and improved timestamps available with WhisperX
- ✅ **Clean Architecture**: Extensible design for future engine additions

## Implementation Strategy

### Phase 1: Architecture Refactoring (2-3 hours)

#### 1.1 Create Abstract Base Engine Class
**File**: `engines/base_whisper_engine.py` (new)
```python
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
from dataclasses import dataclass
import numpy as np

@dataclass
class TranscriptionResult:
    """Standardized transcription result for all engines"""
    chunk_id: int
    text: str
    start_time: float
    end_time: float
    confidence: float
    language: str
    processing_time: float
    segments: List[Dict]
    speaker_labels: Optional[List[str]] = None  # New field for WhisperX

class BaseWhisperEngine(ABC):
    """Abstract base class for all Whisper engines"""
    
    @abstractmethod
    def load_model(self):
        """Load the Whisper model"""
        pass
    
    @abstractmethod
    def start(self):
        """Start the engine and worker threads"""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop the engine and cleanup resources"""
        pass
    
    @abstractmethod
    def submit_chunk(self, chunk) -> bool:
        """Submit audio chunk for transcription"""
        pass
    
    @abstractmethod
    def get_result(self, timeout: Optional[float] = None) -> Optional[TranscriptionResult]:
        """Get transcription result"""
        pass
    
    @abstractmethod
    def transcribe_file(self, audio_file: str, **kwargs) -> List[TranscriptionResult]:
        """Transcribe complete audio file"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict:
        """Get engine statistics"""
        pass
    
    @property
    @abstractmethod
    def device(self) -> str:
        """Current device (cpu/cuda)"""
        pass
```

#### 1.2 Refactor Existing Engine
**Action**: Rename `whisper_engine.py` → `engines/faster_whisper_engine.py`
**Changes**: 
- Inherit from `BaseWhisperEngine`
- Update imports throughout codebase
- Ensure all abstract methods are implemented

#### 1.3 Create Engine Factory
**File**: `engines/engine_factory.py` (new)
```python
from typing import Dict, Any
from .base_whisper_engine import BaseWhisperEngine

def create_whisper_engine(engine_type: str, **kwargs) -> BaseWhisperEngine:
    """Factory function to create appropriate whisper engine"""
    
    if engine_type == "faster-whisper":
        try:
            from .faster_whisper_engine import FasterWhisperEngine
            return FasterWhisperEngine(**kwargs)
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            )
    
    elif engine_type == "whisperx":
        try:
            from .whisperx_engine import WhisperXEngine
            return WhisperXEngine(**kwargs)
        except ImportError:
            raise ImportError(
                "WhisperX not installed. Install with: pip install whisperx\n"
                "For speaker diarization, also run: pip install pyannote-audio\n"
                "Or use --engine faster-whisper (default)"
            )
    
    else:
        available_engines = ["faster-whisper", "whisperx"]
        raise ValueError(f"Unknown engine type: {engine_type}. Available: {available_engines}")

def get_available_engines() -> List[str]:
    """Get list of available engines based on installed packages"""
    available = []
    
    # Check faster-whisper
    try:
        import faster_whisper
        available.append("faster-whisper")
    except ImportError:
        pass
    
    # Check WhisperX
    try:
        import whisperx
        available.append("whisperx")
    except ImportError:
        pass
    
    return available
```

### Phase 2: WhisperX Engine Implementation (3-4 hours)

#### 2.1 Create WhisperX Engine Class
**File**: `engines/whisperx_engine.py` (new)

**Key Implementation Points**:
- Lazy import of WhisperX dependencies
- Map WhisperX API to `BaseWhisperEngine` interface
- Handle speaker diarization as optional feature
- Implement both real-time chunked processing and file transcription
- Error handling for missing Hugging Face tokens

**Core Methods to Implement**:
```python
class WhisperXEngine(BaseWhisperEngine):
    def __init__(self, 
                 model_size: str = "base",
                 device: str = "auto", 
                 language: Optional[str] = None,
                 enable_diarization: bool = False,
                 huggingface_token: Optional[str] = None,
                 batch_size: int = 16,
                 **kwargs):
        # Implementation details
        
    def _setup_diarization(self):
        """Setup speaker diarization pipeline if enabled"""
        
    def _transcribe_with_whisperx(self, audio, **kwargs):
        """Core WhisperX transcription with alignment and diarization"""
        
    def transcribe_file(self, audio_file: str, include_timestamps: bool = False, 
                       enable_diarization: bool = None) -> List[TranscriptionResult]:
        """File transcription with optional speaker diarization"""
```

#### 2.2 Handle WhisperX-Specific Features
- **Speaker Diarization**: Optional feature with clear enable/disable
- **Word-level Alignment**: Improved timestamp accuracy
- **Batch Processing**: Efficient processing for large files
- **Token Management**: Handle Hugging Face authentication

### Phase 3: Configuration Updates (1 hour)

#### 3.1 Update config.yaml Schema
```yaml
# Existing section - fully backward compatible
whisper:
  engine: "faster-whisper"     # NEW: Engine selection
  model_size: "base"          # ✅ Compatible with both engines
  device: "auto"              # ✅ Compatible with both engines  
  language: null              # ✅ Compatible with both engines
  word_timestamps: true       # ✅ Compatible with both engines
  
  # faster-whisper specific settings (ignored by WhisperX)
  compute_type: "float16"     # Only used by faster-whisper
  beam_size: 1               # Only used by faster-whisper
  
# NEW: WhisperX-specific settings
whisperx:
  batch_size: 16             # WhisperX batch processing size
  enable_diarization: false  # Speaker separation
  huggingface_token: null    # HF token for diarization model
  
  # Diarization settings
  diarization:
    min_speakers: null       # Minimum number of speakers (auto-detect if null)
    max_speakers: null       # Maximum number of speakers (auto-detect if null)
```

#### 3.2 Configuration Validation
- Validate engine-specific settings
- Provide warnings for unsupported combinations
- Default fallbacks for missing configurations

### Phase 4: Command Line Interface Updates (1 hour)

#### 4.1 New Command Line Arguments
```python
# Add to argument parser
parser.add_argument('--engine', 
                   choices=['faster-whisper', 'whisperx'],
                   default='faster-whisper',
                   help='Whisper engine to use (default: faster-whisper)')

parser.add_argument('--diarization', 
                   action='store_true',
                   help='Enable speaker diarization (WhisperX only)')

parser.add_argument('--list-engines',
                   action='store_true', 
                   help='List available engines and exit')

parser.add_argument('--hf-token',
                   help='Hugging Face token for diarization (can also set HF_TOKEN env var)')
```

#### 4.2 Argument Validation
- Check engine availability before processing
- Validate engine-specific argument combinations
- Provide helpful error messages

### Phase 5: Application Integration (1-2 hours)

#### 5.1 Update TranscriberApp Class
**File**: `whisper-transcriber.py`

**Key Changes**:
```python
def __init__(self, config_path: str = "config.yaml", 
             output_file: Optional[str] = None, 
             suppress_gui: bool = False,
             engine: str = "faster-whisper"):  # NEW
    
    # Use engine factory instead of direct instantiation
    self.whisper_engine = create_whisper_engine(
        engine_type=engine,
        **self._get_engine_config(engine)
    )
```

#### 5.2 Engine Configuration Mapping
- Map unified config to engine-specific parameters
- Handle engine-specific features (diarization, etc.)
- Maintain backward compatibility

### Phase 6: Batch Script Integration (30 minutes)

#### 6.1 Update batch-transcribe.ps1
**New Parameters**:
```powershell
param(
    # ... existing parameters ...
    [ValidateSet('faster-whisper', 'whisperx')]
    [string]$Engine = 'faster-whisper',
    [switch]$Diarization,
    [string]$HuggingFaceToken
)
```

**Updated Help Text**:
```powershell
Options:
  -Engine <engine>      Whisper engine (faster-whisper/whisperx, default: faster-whisper)
  -Diarization          Enable speaker separation (WhisperX only)  
  -HuggingFaceToken     HF token for diarization
```

#### 6.2 Parameter Validation
- Check if WhisperX is available when requested
- Validate diarization requirements
- Pass engine selection to Python script

### Phase 7: Testing & Documentation (2-3 hours)

#### 7.1 Create Test Plan
**Files to Create**:
- `tests/test_engine_factory.py`
- `tests/test_whisperx_engine.py` 
- `tests/test_backward_compatibility.py`

**Test Scenarios**:
- ✅ Faster-whisper engine works unchanged
- ✅ WhisperX engine basic transcription
- ✅ Speaker diarization functionality
- ✅ Engine fallback when dependencies missing
- ✅ Configuration validation
- ✅ Command line argument processing

#### 7.2 Update Documentation
**README.md Updates**:
- Add WhisperX installation instructions
- Document new command line options
- Add speaker diarization examples
- Update troubleshooting section

**New Documentation**:
- Engine comparison table (performance, features, requirements)
- WhisperX configuration guide  
- Speaker diarization setup instructions

## Detailed Implementation Steps

### Step 1: Environment Preparation
```bash
# Create new branch
git checkout -b feature/whisperx-engine

# Create engines directory
mkdir engines
touch engines/__init__.py

# Install WhisperX for testing (optional)
pip install whisperx
```

### Step 2: Architecture Implementation
1. Create `engines/base_whisper_engine.py`
2. Move and refactor `whisper_engine.py` → `engines/faster_whisper_engine.py`
3. Create `engines/engine_factory.py`
4. Update all imports in main files

### Step 3: WhisperX Engine Development
1. Create `engines/whisperx_engine.py`
2. Implement all abstract methods
3. Add speaker diarization support
4. Test basic functionality

### Step 4: Integration & Configuration
1. Update `config.yaml` schema
2. Update `TranscriberApp` to use factory
3. Add command line arguments
4. Update batch script

### Step 5: Testing & Validation
1. Test backward compatibility
2. Test WhisperX functionality
3. Test error handling
4. Performance comparison

## Command Line Examples

```bash
# Current behavior (unchanged)
python whisper-transcriber.py -i audio.wav -o transcript.txt

# Use WhisperX engine
python whisper-transcriber.py --engine whisperx -i audio.wav -o transcript.txt

# WhisperX with speaker diarization  
python whisper-transcriber.py --engine whisperx --diarization -i conversation.wav -o transcript.txt

# List available engines
python whisper-transcriber.py --list-engines

# Batch processing with WhisperX
.\batch-transcribe.ps1 -Engine whisperx -Diarization -Model medium -Timestamps
```

## Expected Output Formats

### Faster-whisper (unchanged)
```
[00:00:00] I don't know. What are you doing?
[00:00:05] I'm just going to ride a deer.
```

### WhisperX without diarization
```
[00:00:00] I don't know. What are you doing?
[00:00:05] I'm just going to ride a deer.
```

### WhisperX with diarization
```
[00:00:00] Speaker A: I don't know. What are you doing?
[00:00:05] Speaker B: I'm just going to ride a deer.
[00:00:12] Speaker A: Reinforcing stereotypes?
```

## Error Handling Scenarios

### WhisperX Not Installed
```
❌ WhisperX not installed!
Install with: pip install whisperx
For speaker diarization: pip install pyannote-audio  
Or use: --engine faster-whisper
```

### Missing Hugging Face Token (for diarization)
```
⚠️ Diarization requested but no Hugging Face token provided
Set HF_TOKEN environment variable or use --hf-token argument
Diarization models require authentication for download
Proceeding without speaker diarization...
```

### Invalid Engine Selection
```
❌ Unknown engine type: whisper-cpp
Available engines: faster-whisper, whisperx
Use --list-engines to see available engines
```

## Performance Considerations

### Memory Usage
- **faster-whisper**: Lower baseline memory, good for real-time
- **WhisperX**: Higher memory usage, especially with diarization
- Implement memory monitoring and warnings

### Processing Speed  
- **faster-whisper**: Optimized for streaming/real-time
- **WhisperX**: Optimized for batch processing, better accuracy
- Add processing time comparisons to stats

### GPU Resources
- Both engines support CUDA
- WhisperX may use more VRAM for diarization models
- Implement graceful fallback to CPU

## Dependencies

### Core Dependencies (unchanged)
```txt
faster-whisper>=0.10.0
librosa>=0.10.1
soundfile>=0.12.1
torch>=2.0.0
# ... existing deps
```

### Optional Dependencies
```txt
# For WhisperX support
whisperx>=3.1.0
pyannote-audio>=3.1.0
```

### Installation Options
```bash
# Basic installation (faster-whisper only)
pip install -r requirements.txt

# Full installation (both engines)
pip install -r requirements.txt -r requirements-whisperx.txt

# Or with extras (future setup.py)
pip install whisper-transcriber[whisperx]
```

## Risks & Mitigation

### Risk: Breaking Changes
**Mitigation**: Extensive backward compatibility testing, default to faster-whisper

### Risk: Complex Dependencies  
**Mitigation**: Optional imports, clear error messages, graceful fallbacks

### Risk: Performance Regression
**Mitigation**: Performance benchmarks, memory monitoring, user choice

### Risk: Configuration Complexity
**Mitigation**: Sensible defaults, validation, clear documentation

## Success Criteria

- ✅ All existing functionality works unchanged with `--engine faster-whisper` (default)
- ✅ WhisperX engine provides transcription with `--engine whisperx` 
- ✅ Speaker diarization works with `--diarization` flag
- ✅ Program runs without WhisperX installed (when using faster-whisper)
- ✅ Clear error messages guide users to install missing dependencies
- ✅ Batch processing script supports both engines
- ✅ Configuration remains backward compatible
- ✅ Documentation covers all new features
- ✅ Performance is equivalent or better for respective use cases

## Future Extensions

This architecture enables future additions:
- **OpenAI API Engine**: For cloud-based processing
- **Whisper.cpp Engine**: For CPU-optimized processing  
- **Custom Models**: Support for fine-tuned models
- **Streaming Engines**: Optimized for real-time processing

---

**Estimated Total Development Time: 8-12 hours**
**Complexity Level: Medium**
**Risk Level: Low** (due to backward compatibility focus)
