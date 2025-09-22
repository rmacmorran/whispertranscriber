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

from rich.console import Console
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
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the transcriber application
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
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
        
        # Output
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
                'max_log_size': 10
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
                console.print(f"[green]✅ Configuration loaded from {config_file}[/green]")
                
            except Exception as e:
                console.print(f"[yellow]⚠️  Failed to load config: {e}. Using defaults.[/yellow]")
        else:
            console.print(f"[yellow]⚠️  Config file {config_file} not found. Using defaults.[/yellow]")
        
        return default_config
    
    def setup_logging(self):
        """
        Setup logging configuration
        """
        log_level = getattr(logging, self.config['performance']['log_level'].upper())
        
        # Setup file logging
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'transcriber.log'),
                logging.StreamHandler(sys.stdout) if log_level == logging.DEBUG else logging.NullHandler()
            ]
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
                console.print("\n[yellow]Received interrupt signal, shutting down...[/yellow]")
                self.shutdown()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Main display loop
            console.print("[green]✅ Transcription started! Press Ctrl+C to stop.[/green]\n")
            
            with Live(self.create_display(), refresh_per_second=2, screen=True) as live:
                while not self.shutdown_event.is_set():
                    live.update(self.create_display())
                    time.sleep(0.5)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            logger.error(f"Application error: {e}")
            console.print(f"[red]❌ Application error: {e}[/red]")
        finally:
            self.shutdown()
    
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
  python main.py                    # Run with default settings
  python main.py --list-devices     # Show available audio devices
  python main.py --device 31        # Use specific audio device
  python main.py --model small      # Use small model for better accuracy
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
    
    args = parser.parse_args()
    
    # Handle list devices
    if args.list_devices:
        print_audio_devices()
        return
    
    # Create and configure app
    try:
        app = TranscriberApp(config_path=args.config)
        
        # Apply command line overrides
        if args.device is not None:
            app.config['audio']['device_index'] = args.device
        
        if args.model:
            app.config['whisper']['model_size'] = args.model
        
        if args.language:
            app.config['whisper']['language'] = args.language if args.language != 'auto' else None
        
        # Run the application
        app.run()
        
    except Exception as e:
        console.print(f"[red]❌ Failed to start application: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
