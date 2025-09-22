#!/usr/bin/env python3
"""
Real-time Whisper Audio Transcriber

A near-realtime audio transcription tool using faster-whisper and VAD-based chunking
"""

import sys
import time
import signal
import logging
import argparse
import threading
from pathlib import Path
from typing import Optional, Dict
import yaml
import pynvml
import librosa
import soundfile as sf

from rich.console import Console
import os
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from audio_devices import list_audio_devices, print_audio_devices, find_device_by_name
from audio_buffer import ContinuousAudioBuffer
from vad_chunker import VADChunker
from whisper_engine import WhisperEngine, TranscriptionResult
import numpy as np

console = Console()

class TranscriberApp:
    """
    Main application class for real-time audio transcription
    """
    
    def __init__(self, config_path: str = "config.yaml", output_file: Optional[str] = None, suppress_gui: bool = False):
        """
        Initialize the transcriber application
        
        Args:
            config_path: Path to configuration file
            output_file: Optional output file path for saving transcriptions
            suppress_gui: Whether to suppress the GUI and run in console mode
        """
        self.config_path = config_path
        
        # Output configuration (set before load_config to avoid AttributeError)
        self.output_file = output_file
        self.suppress_gui = suppress_gui
        
        self.config = self.load_config()
        
        # Components
        self.audio_buffer: Optional[ContinuousAudioBuffer] = None
        self.vad_chunker: Optional[VADChunker] = None
        self.whisper_engine: Optional[WhisperEngine] = None
        
        # State
        self.is_running = False
        self.start_time = None
        self.shutdown_event = threading.Event()
        self.transcription_thread = None
        self.output_file_handle = None
        self.transcript_lines = []  # Store transcription lines for display
        self.max_transcript_lines = 50  # Maximum lines to keep in memory
        
        # Statistics
        self.stats = {
            'chunks_received': 0,
            'chunks_transcribed': 0,
            'total_audio_duration': 0,
            'words_transcribed': 0
        }
        
        # GPU monitoring
        self.gpu_available = False
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
        except:
            pass
        
        # Setup logging
        self.setup_logging()
        
        logger = logging.getLogger(__name__)
        logger.info("Whisper Real-time Transcriber initialized")
    
    def load_config(self) -> Dict:
        """
        Load configuration from YAML file with fallback to defaults
        
        Returns:
            Configuration dictionary
        """
        default_config = {
            'audio': {
                'device_index': -1,
                'sample_rate': 16000,
                'buffer_size_ms': 20,
                'max_buffer_duration': 30
            },
            'vad': {
                'threshold': 0.5,
                'min_silence_ms': 300,
                'min_chunk_duration': 2.0,
                'max_chunk_duration': 12.0,
                'chunk_overlap': 0.5
            },
            'whisper': {
                'model_size': 'base',
                'device': 'auto',
                'compute_type': 'float16',
                'beam_size': 1,
                'language': None,
                'word_timestamps': True
            },
            'output': {
                'console_output': True,
                'file_formats': [],
                'output_dir': './transcripts',
                'include_confidence': True
            },
            'performance': {
                'num_workers': 2,
                'monitor_gpu': True,
                'log_level': 'INFO',
                'max_log_size': 10,
                'enable_file_logging': False  # Set to True to enable transcriber.log
            }
        }
        
        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                
                # Merge with defaults (deep merge)
                def merge_dict(base, override):
                    for key, value in override.items():
                        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                            merge_dict(base[key], value)
                        else:
                            base[key] = value
                
                merge_dict(default_config, loaded_config)
                # Use safe console output that avoids Unicode issues
                if self.suppress_gui:
                    print(f"Configuration loaded from {config_file}")
                else:
                    try:
                        console.print(f"[green]✅ Configuration loaded from {config_file}[/green]")
                    except UnicodeEncodeError:
                        print(f"Configuration loaded from {config_file}")
                
            except Exception as e:
                if self.suppress_gui:
                    print(f"Failed to load config: {e}. Using defaults.")
                else:
                    try:
                        console.print(f"[yellow]⚠️  Failed to load config: {e}. Using defaults.[/yellow]")
                    except UnicodeEncodeError:
                        print(f"Failed to load config: {e}. Using defaults.")
        else:
            if self.suppress_gui:
                print(f"Config file {config_file} not found. Using defaults.")
            else:
                try:
                    console.print(f"[yellow]⚠️  Config file {config_file} not found. Using defaults.[/yellow]")
                except UnicodeEncodeError:
                    print(f"Config file {config_file} not found. Using defaults.")
        
        return default_config
    
    def setup_logging(self):
        """
        Setup logging configuration
        """
        log_level = getattr(logging, self.config['performance']['log_level'].upper())
        
        # Prepare handlers list
        handlers = []
        
        # Add file handler if enabled
        if self.config['performance'].get('enable_file_logging', False):
            # Determine log file directory
            if self.config['output'].get('output_dir'):
                # Use output directory if specified
                log_dir = Path(self.config['output']['output_dir'])
            else:
                # Use current working directory
                log_dir = Path('.')
            
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file_path = log_dir / 'transcriber.log'
            handlers.append(logging.FileHandler(log_file_path))
        
        # Add console handler for DEBUG level
        if log_level == logging.DEBUG:
            handlers.append(logging.StreamHandler(sys.stdout))
        else:
            handlers.append(logging.NullHandler())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )
    
    def resolve_audio_device(self) -> Optional[int]:
        """
        Resolve audio device from configuration
        
        Returns:
            Device ID or None for default
        """
        device_index = self.config['audio']['device_index']
        
        if device_index == -1:
            return None  # Use system default
        
        # Check if device exists
        devices = list_audio_devices()
        device_ids = [d['id'] for d in devices]
        
        if device_index in device_ids:
            return device_index
        else:
            console.print(f"[yellow]⚠️  Device {device_index} not found. Using system default.[/yellow]")
            return None
    
    def initialize_components(self):
        """
        Initialize audio buffer, VAD chunker, and Whisper engine
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Resolve audio device
            device_id = self.resolve_audio_device()
            
            # Initialize audio buffer
            self.audio_buffer = ContinuousAudioBuffer(
                device_id=device_id,
                sample_rate=self.config['audio']['sample_rate'],
                channels=1,
                buffer_size_ms=self.config['audio']['buffer_size_ms'],
                max_buffer_duration=self.config['audio']['max_buffer_duration']
            )
            
            # Initialize VAD chunker
            self.vad_chunker = VADChunker(
                sample_rate=self.config['audio']['sample_rate'],
                min_silence_ms=self.config['vad']['min_silence_ms'],
                min_chunk_duration=self.config['vad']['min_chunk_duration'],
                max_chunk_duration=self.config['vad']['max_chunk_duration'],
                chunk_overlap=self.config['vad']['chunk_overlap'],
                vad_threshold=self.config['vad']['threshold']
            )
            
            # Initialize Whisper engine
            self.whisper_engine = WhisperEngine(
                model_size=self.config['whisper']['model_size'],
                device=self.config['whisper']['device'],
                compute_type=self.config['whisper']['compute_type'],
                num_workers=self.config['performance']['num_workers'],
                beam_size=self.config['whisper']['beam_size'],
                language=self.config['whisper']['language'],
                word_timestamps=self.config['whisper']['word_timestamps']
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def transcription_worker(self):
        """
        Worker thread that handles the transcription pipeline
        """
        logger = logging.getLogger(__name__)
        logger.info("Transcription worker started")
        
        last_vad_time = 0
        vad_interval = 0.1  # Process VAD every 100ms
        
        while not self.shutdown_event.is_set():
            try:
                current_time = time.time() - self.start_time
                
                # Process VAD at regular intervals
                if current_time - last_vad_time >= vad_interval:
                    # Get new audio from buffer
                    new_audio_duration = current_time - last_vad_time
                    new_audio = self.audio_buffer.get_latest(new_audio_duration)
                    
                    if len(new_audio) > 0:
                        # Process through VAD chunker
                        chunks = self.vad_chunker.process_audio(new_audio, current_time)
                        
                        # Submit chunks to Whisper engine
                        for chunk in chunks:
                            if chunk.contains_speech:  # Only process speech chunks
                                success = self.whisper_engine.submit_chunk(chunk)
                                if success:
                                    self.stats['chunks_received'] += 1
                                    self.stats['total_audio_duration'] += chunk.duration
                    
                    last_vad_time = current_time
                
                # Check for transcription results
                result = self.whisper_engine.get_result(timeout=0.05)  # Short timeout
                if result:
                    self.handle_transcription_result(result)
                
            except Exception as e:
                logger.error(f"Transcription worker error: {e}")
                time.sleep(0.1)
        
        # Finalize remaining chunks
        try:
            final_chunk = self.vad_chunker.finalize_current_chunk()
            if final_chunk and final_chunk.contains_speech:
                self.whisper_engine.submit_chunk(final_chunk)
                
            # Process any remaining results
            while True:
                result = self.whisper_engine.get_result(timeout=1.0)
                if result is None:
                    break
                self.handle_transcription_result(result)
                
        except Exception as e:
            logger.error(f"Error finalizing transcription: {e}")
        
        logger.info("Transcription worker stopped")
    
    def handle_transcription_result(self, result: TranscriptionResult):
        """
        Handle a transcription result
        
        Args:
            result: Transcription result from Whisper
        """
        if result.text.strip():
            # Format timestamp
            timestamp = time.strftime("%H:%M:%S", time.localtime(self.start_time + result.start_time))
            
            # Create transcript line
            confidence_str = f" ({result.confidence:.2f})" if self.config['output']['include_confidence'] else ""
            line = f"[{timestamp}] {result.text.strip()}{confidence_str}"
            
            # Add to transcript lines
            self.transcript_lines.append(line)
            if len(self.transcript_lines) > self.max_transcript_lines:
                self.transcript_lines.pop(0)
            
            # Write to output file if specified
            if self.output_file and self.output_file_handle:
                try:
                    self.output_file_handle.write(line + "\n")
                    self.output_file_handle.flush()  # Ensure immediate write
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.error(f"Failed to write to output file: {e}")
            
            # Console output for non-GUI mode
            if self.suppress_gui:
                try:
                    print(line)
                except UnicodeEncodeError:
                    # If line contains chars that can't be encoded, encode safely
                    safe_line = line.encode('ascii', errors='replace').decode('ascii')
                    print(safe_line)
            
            # Update statistics
            self.stats['chunks_transcribed'] += 1
            self.stats['words_transcribed'] += len(result.text.split())
            
            # Log the transcription
            logger = logging.getLogger(__name__)
            logger.info(f"Transcribed chunk {result.chunk_id}: {result.text.strip()}")
    
    def get_gpu_stats(self) -> Optional[Dict]:
        """
        Get GPU statistics if available
        
        Returns:
            GPU statistics or None if not available
        """
        if not self.gpu_available:
            return None
        
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
            return {
                'memory_used_mb': mem_info.used // (1024 * 1024),
                'memory_total_mb': mem_info.total // (1024 * 1024),
                'memory_percent': (mem_info.used / mem_info.total) * 100,
                'gpu_utilization': utilization.gpu,
                'memory_utilization': utilization.memory
            }
        except:
            return None
    
    def create_display(self) -> Panel:
        """
        Create the main display panel
        
        Returns:
            Rich Panel with current status
        """
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="transcript", size=30),
            Layout(name="stats", size=12)
        )
        
        # Header section
        uptime = time.time() - self.start_time if self.start_time else 0
        header_text = Text()
        header_text.append("🎤 Real-time Whisper Transcriber\n\n", style="bold blue")
        header_text.append(f"Status: ", style="bold")
        header_text.append("🟢 Running\n" if self.is_running else "🔴 Stopped\n", 
                          style="green" if self.is_running else "red")
        header_text.append(f"Uptime: {uptime:.1f}s\n")
        header_text.append(f"Model: {self.config['whisper']['model_size']} ", style="cyan")
        header_text.append(f"Device: {self.whisper_engine.device if self.whisper_engine else 'Unknown'}\n", style="cyan")
        
        layout["header"].update(Panel(header_text, title="System Status"))
        
        # Transcript section
        if self.transcript_lines:
            transcript_text = Text()
            for line in self.transcript_lines[-25:]:  # Show last 25 lines
                transcript_text.append(line + "\n")
        else:
            transcript_text = Text("Waiting for speech...", style="dim")
        
        layout["transcript"].update(Panel(transcript_text, title="Live Transcript", border_style="green"))
        
        # Statistics section
        stats_table = Table(title="Performance Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        # Audio stats
        if self.audio_buffer:
            buffer_stats = self.audio_buffer.get_stats()
            stats_table.add_row("Audio Duration", f"{buffer_stats['buffer_duration_seconds']:.1f}s")
            stats_table.add_row("Audio Frames", f"{buffer_stats['buffer_frames']:,}")
        
        # Transcription stats
        stats_table.add_row("Chunks Received", str(self.stats['chunks_received']))
        stats_table.add_row("Chunks Transcribed", str(self.stats['chunks_transcribed']))
        stats_table.add_row("Words Transcribed", str(self.stats['words_transcribed']))
        stats_table.add_row("Audio Processed", f"{self.stats['total_audio_duration']:.1f}s")
        
        # Whisper engine stats
        if self.whisper_engine:
            engine_stats = self.whisper_engine.get_stats()
            avg_rtf = engine_stats['avg_processing_time'] / (engine_stats['avg_processing_time'] + 1) if engine_stats['avg_processing_time'] > 0 else 0
            stats_table.add_row("Avg Processing Time", f"{engine_stats['avg_processing_time']:.3f}s")
            stats_table.add_row("Queue Size", f"{engine_stats['chunks_pending']}")
        
        # GPU stats
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            stats_table.add_row("GPU Memory", f"{gpu_stats['memory_used_mb']:,} / {gpu_stats['memory_total_mb']:,} MB")
            stats_table.add_row("GPU Utilization", f"{gpu_stats['gpu_utilization']}%")
        
        layout["stats"].update(Panel(stats_table, title="Statistics"))
        
        return Panel(layout, title="Whisper Real-time Transcriber", border_style="blue")
    
    def run(self):
        """
        Main run method
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Open output file if specified
            if self.output_file:
                try:
                    # Create directory if it doesn't exist
                    output_path = Path(self.output_file)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    self.output_file_handle = open(self.output_file, 'w', encoding='utf-8')
                    console.print(f"[green]✅ Output file opened: {self.output_file}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to open output file: {e}[/red]")
                    raise
            
            # Initialize components
            console.print("[yellow]Initializing components...[/yellow]")
            self.initialize_components()
            
            # Start components
            console.print("[yellow]Starting audio capture...[/yellow]")
            self.audio_buffer.start()
            
            console.print("[yellow]Starting Whisper engine...[/yellow]")
            self.whisper_engine.start()
            
            # Start transcription worker thread
            self.is_running = True
            self.start_time = time.time()
            self.transcription_thread = threading.Thread(
                target=self.transcription_worker,
                name="TranscriptionWorker",
                daemon=True
            )
            self.transcription_thread.start()
            
            # Setup signal handlers
            def signal_handler(signum, frame):
                if self.suppress_gui:
                    print("\nReceived interrupt signal, shutting down...")
                else:
                    console.print("\n[yellow]Received interrupt signal, shutting down...[/yellow]")
                self.shutdown()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Main display loop
            if self.suppress_gui:
                # Console mode - simple text output
                print("Transcription started! Press Ctrl+C to stop.")
                print(f"Model: {self.config['whisper']['model_size']}, Device: {self.whisper_engine.device}")
                if self.output_file:
                    print(f"Writing to: {self.output_file}")
                print("Listening for audio...\n")
                
                # Simple wait loop for console mode
                while not self.shutdown_event.is_set():
                    time.sleep(0.5)
            else:
                # GUI mode with rich interface
                console.print("[green]✅ Transcription started! Press Ctrl+C to stop.[/green]\n")
                
                with Live(self.create_display(), refresh_per_second=2, screen=True) as live:
                    while not self.shutdown_event.is_set():
                        live.update(self.create_display())
                        time.sleep(0.5)
            
        except KeyboardInterrupt:
            if self.suppress_gui:
                print("\nInterrupted by user")
            else:
                console.print("\n[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            logger.error(f"Application error: {e}")
            if self.suppress_gui:
                print(f"Application error: {e}")
            else:
                console.print(f"[red]❌ Application error: {e}[/red]")
        finally:
            self.shutdown()
    
    def transcribe_file(self, input_file: str, include_timestamps: bool = False) -> str:
        """
        Transcribe an audio or video file and return the full transcript
        
        Args:
            input_file: Path to the input audio/video file
            include_timestamps: Whether to include timestamps in the output
            
        Returns:
            Full transcript text
        """
        logger = logging.getLogger(__name__)
        
        try:
            # Validate input file
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            # Load audio file using librosa (handles many formats including video)
            if not self.suppress_gui:
                console.print(f"[yellow]Loading audio from: {input_file}[/yellow]")
            else:
                print(f"Loading audio from: {input_file}")
            
            # Try to load audio and resample to 16kHz for Whisper
            # Use multiple fallback methods for better compatibility
            audio = None
            sr = None
            duration = 0
            
            # Method 1: Try librosa with soundfile backend (preferred)
            try:
                import soundfile as sf
                import numpy as np
                audio, sr = librosa.load(input_file, sr=16000, mono=True)
                duration = len(audio) / sr
                if not self.suppress_gui:
                    console.print(f"[green]✅ Loaded with librosa+soundfile backend[/green]")
                else:
                    print("Loaded with librosa+soundfile backend")
            except Exception as e1:
                if not self.suppress_gui:
                    console.print(f"[yellow]⚠️ Librosa+soundfile failed: {str(e1)[:100]}...[/yellow]")
                else:
                    print(f"Librosa+soundfile failed: {str(e1)[:100]}...")
                
                # Method 2: Try librosa with audioread backend
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        audio, sr = librosa.load(input_file, sr=16000, mono=True)
                        duration = len(audio) / sr
                    if not self.suppress_gui:
                        console.print(f"[green]✅ Loaded with librosa+audioread backend[/green]")
                    else:
                        print("Loaded with librosa+audioread backend")
                except Exception as e2:
                    if not self.suppress_gui:
                        console.print(f"[yellow]⚠️ Librosa+audioread failed: {str(e2)[:100]}...[/yellow]")
                    else:
                        print(f"Librosa+audioread failed: {str(e2)[:100]}...")
                    
                    # Method 3: Try direct soundfile loading
                    try:
                        import soundfile as sf
                        import numpy as np
                        from scipy import signal
                        
                        audio_data, original_sr = sf.read(input_file, always_2d=False)
                        if len(audio_data.shape) > 1:  # Convert stereo to mono
                            audio_data = np.mean(audio_data, axis=1)
                        
                        # Resample to 16kHz if needed
                        if original_sr != 16000:
                            from scipy.signal import resample
                            num_samples = int(len(audio_data) * 16000 / original_sr)
                            audio = resample(audio_data, num_samples).astype(np.float32)
                        else:
                            audio = audio_data.astype(np.float32)
                        
                        sr = 16000
                        duration = len(audio) / sr
                        if not self.suppress_gui:
                            console.print(f"[green]✅ Loaded with direct soundfile[/green]")
                        else:
                            print("Loaded with direct soundfile")
                    except Exception as e3:
                        # All methods failed - provide comprehensive error message
                        error_msg = f"Failed to load audio from file using all available methods:\n"
                        error_msg += f"  1. Librosa+SoundFile: {str(e1)[:100]}...\n"
                        error_msg += f"  2. Librosa+AudioRead: {str(e2)[:100]}...\n"
                        error_msg += f"  3. Direct SoundFile: {str(e3)[:100]}...\n\n"
                        
                        file_ext = input_file.lower().split('.')[-1]
                        if file_ext in ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'm4v']:
                            error_msg += "This is a video file. Possible solutions:\n"
                            error_msg += "  • Install/update FFmpeg: winget install ffmpeg\n"
                            error_msg += "  • Try converting to audio first: ffmpeg -i input.mp4 -acodec pcm_s16le output.wav\n"
                        else:
                            error_msg += "This is an audio file. Possible solutions:\n"
                            error_msg += "  • File may be corrupted - try re-downloading\n"
                            error_msg += "  • Unsupported codec - try converting: ffmpeg -i input.ext output.wav\n"
                        
                        error_msg += "  • Check file exists and is readable\n"
                        error_msg += f"  • File extension: .{file_ext}\n"
                        raise Exception(error_msg)
            
            # Validate loaded audio
            if audio is None or len(audio) == 0:
                raise Exception("Audio file loaded but contains no audio data")
            
            # Check for extremely short files
            if duration < 0.1:
                raise Exception(f"Audio file is too short ({duration:.3f}s) - minimum 0.1 seconds required")
            
            if not self.suppress_gui:
                console.print(f"[green]✅ Audio loaded: {duration:.1f}s duration[/green]")
            else:
                print(f"Audio loaded: {duration:.1f}s duration")
            
            # Initialize only Whisper engine for file transcription
            self.whisper_engine = WhisperEngine(
                model_size=self.config['whisper']['model_size'],
                device=self.config['whisper']['device'],
                compute_type=self.config['whisper']['compute_type'],
                num_workers=1,  # Use single worker for file processing
                beam_size=self.config['whisper']['beam_size'],
                language=self.config['whisper']['language'],
                word_timestamps=self.config['whisper']['word_timestamps']
            )
            
            # Start Whisper engine
            self.whisper_engine.start()
            
            # Process file in chunks to handle large files
            chunk_duration = 30.0  # Process 30-second chunks
            chunk_samples = int(chunk_duration * sr)
            total_chunks = int(np.ceil(len(audio) / chunk_samples))
            
            if not self.suppress_gui:
                console.print(f"[yellow]Processing {total_chunks} chunks...[/yellow]")
            else:
                print(f"Processing {total_chunks} chunks...")
            
            transcript_parts = []
            
            for i in range(total_chunks):
                start_idx = i * chunk_samples
                end_idx = min((i + 1) * chunk_samples, len(audio))
                chunk_audio = audio[start_idx:end_idx]
                
                # Convert to int16 format expected by AudioChunk
                chunk_audio_int16 = (chunk_audio * 32767).astype(np.int16)
                
                # Create AudioChunk
                from vad_chunker import AudioChunk
                chunk = AudioChunk(
                    audio_data=chunk_audio_int16,
                    start_time=start_idx / sr,
                    end_time=end_idx / sr,
                    duration=(end_idx - start_idx) / sr,
                    contains_speech=True,  # Assume all chunks contain speech for file mode
                    chunk_id=i,
                    sample_rate=16000
                )
                
                # Submit chunk and wait for result
                success = self.whisper_engine.submit_chunk(chunk)
                if success:
                    # Wait for transcription result
                    result = None
                    timeout_start = time.time()
                    while result is None and (time.time() - timeout_start) < 30.0:
                        result = self.whisper_engine.get_result(timeout=1.0)
                    
                    if result and result.text.strip():
                        # Format with or without timestamps based on parameter
                        if include_timestamps:
                            # Format timestamp from chunk start time
                            start_seconds = int(chunk.start_time)
                            start_minutes = start_seconds // 60
                            start_hours = start_minutes // 60
                            start_seconds = start_seconds % 60
                            start_minutes = start_minutes % 60
                            
                            timestamp_str = f"[{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d}] "
                            formatted_text = timestamp_str + result.text.strip()
                        else:
                            formatted_text = result.text.strip()
                        
                        transcript_parts.append(formatted_text)
                        
                        # Show progress
                        if not self.suppress_gui:
                            progress_msg = f"Chunk {i+1}/{total_chunks}: {result.text.strip()[:50]}..."
                            console.print(f"[cyan]{progress_msg}[/cyan]")
                        else:
                            # For suppress_gui mode, encode transcript text safely for console output
                            try:
                                transcript_preview = result.text.strip()[:50]
                                progress_msg = f"Chunk {i+1}/{total_chunks}: {transcript_preview}..."
                                print(progress_msg)
                            except UnicodeEncodeError:
                                # If transcript contains chars that can't be encoded, show without preview
                                progress_msg = f"Chunk {i+1}/{total_chunks}: [text contains special characters]..."
                                print(progress_msg)
            
            # Stop Whisper engine
            self.whisper_engine.stop()
            
            # Combine all transcript parts
            if include_timestamps:
                # For timestamps, join with single newlines (each chunk is already formatted as one line)
                full_transcript = '\n'.join(transcript_parts)
            else:
                # For no timestamps, join with single space as before
                full_transcript = ' '.join(transcript_parts)
            
            if not self.suppress_gui:
                console.print(f"[green]✅ File transcription completed![/green]")
            else:
                print("File transcription completed!")
            
            return full_transcript
            
        except Exception as e:
            logger.error(f"File transcription error: {e}")
            if not self.suppress_gui:
                console.print(f"[red]❌ File transcription failed: {e}[/red]")
            else:
                print(f"File transcription failed: {e}")
            raise
    
    def shutdown(self):
        """
        Clean shutdown of all components
        """
        if not self.is_running:
            return
        
        logger = logging.getLogger(__name__)
        logger.info("Shutting down application...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop transcription worker
        if self.transcription_thread and self.transcription_thread.is_alive():
            self.transcription_thread.join(timeout=2.0)
        
        # Stop components
        if self.whisper_engine:
            self.whisper_engine.stop()
        
        if self.audio_buffer:
            self.audio_buffer.stop()
        
        # Close output file if open
        if self.output_file_handle:
            try:
                self.output_file_handle.close()
                if self.suppress_gui:
                    print(f"Transcription saved to: {self.output_file}")
                else:
                    console.print(f"[green]Transcription saved to: {self.output_file}[/green]")
            except Exception as e:
                logger.error(f"Failed to close output file: {e}")
        
        logger.info("Application shutdown complete")


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription using Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time transcription (default mode)
  python main.py                              # Run with default settings (GUI mode)
  python main.py --list-devices               # Show available audio devices
  python main.py --device 31                  # Use specific audio device
  python main.py --model small                # Use small model for better accuracy
  python main.py --output transcript.txt      # Save transcription to file
  python main.py --no-gui                     # Run in console mode without GUI
  python main.py -q                           # Run in console mode (short form)
  python main.py -d 31 -o output.txt -q       # Console mode with device and output file
  
  # File transcription mode
  python main.py -i audio.mp3                 # Transcribe audio file to console
  python main.py -i video.mp4 -o transcript.txt # Transcribe video file to text file
  python main.py -i recording.wav -q          # Transcribe in console mode
  python main.py -i audio.m4a -m small --language en # Use small model with English language
  python main.py -i video.mp4 -t -o transcript.txt # Include timestamps in output
  python main.py -i video.mp4 -tq             # Transcribe with timestamps, console mode
        """
    )
    
    parser.add_argument('--config', '-c', 
                       default='config.yaml',
                       help='Configuration file path (default: config.yaml)')
    
    parser.add_argument('--list-devices', '-l',
                       action='store_true',
                       help='List available audio input devices and exit')
    
    parser.add_argument('--device', '-d',
                       type=int,
                       help='Audio input device ID (overrides config)')
    
    parser.add_argument('--model', '-m',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
                       help='Whisper model size (overrides config)')
    
    parser.add_argument('--language',
                       help='Language code (e.g., en, es, fr) or auto for detection')
    
    parser.add_argument('--output', '-o',
                       help='Output file path for saving transcriptions')
    
    parser.add_argument('--no-gui', '--suppress-gui', '-q',
                       action='store_true',
                       help='Suppress GUI and run in console mode')
    
    parser.add_argument('--input-file', '-i',
                       type=str,
                       help='Input audio/video file to transcribe (non-real-time mode)')
    
    parser.add_argument('--timestamps', '-t',
                       action='store_true',
                       help='Include timestamps in file transcription output')
    
    args = parser.parse_args()
    
    # Handle list devices
    if args.list_devices:
        print_audio_devices()
        return
    
    # Create and configure app
    try:
        app = TranscriberApp(
            config_path=args.config,
            output_file=args.output,
            suppress_gui=args.no_gui
        )
        
        # Apply command line overrides
        if args.device is not None:
            app.config['audio']['device_index'] = args.device
        
        if args.model:
            app.config['whisper']['model_size'] = args.model
        
        if args.language:
            app.config['whisper']['language'] = args.language if args.language != 'auto' else None
        
        # Check if we're in file transcription mode
        if args.input_file:
            # File transcription mode
            try:
                transcript = app.transcribe_file(args.input_file, include_timestamps=args.timestamps)
                
                # Output the transcript
                if args.output:
                    # Write to output file
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(args.output, 'w', encoding='utf-8') as f:
                        f.write(transcript)
                    
                    if args.no_gui:
                        print(f"\nTranscript saved to: {args.output}")
                    else:
                        try:
                            console.print(f"\n[green]✅ Transcript saved to: {args.output}[/green]")
                        except UnicodeEncodeError:
                            print(f"\nTranscript saved to: {args.output}")
                else:
                    # Output to console
                    if args.no_gui:
                        print("\n--- TRANSCRIPT ---")
                        print(transcript)
                        print("--- END TRANSCRIPT ---")
                    else:
                        console.print("\n[bold green]--- TRANSCRIPT ---[/bold green]")
                        console.print(transcript)
                        console.print("[bold green]--- END TRANSCRIPT ---[/bold green]")
                
            except Exception as e:
                if args.no_gui:
                    print(f"File transcription failed: {e}")
                else:
                    try:
                        console.print(f"[red]❌ File transcription failed: {e}[/red]")
                    except UnicodeEncodeError:
                        print(f"File transcription failed: {e}")
                sys.exit(1)
        else:
            # Real-time transcription mode
            app.run()
        
    except Exception as e:
        try:
            console.print(f"[red]❌ Failed to start application: {e}[/red]")
        except UnicodeEncodeError:
            print(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
