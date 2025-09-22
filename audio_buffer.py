#!/usr/bin/env python3
"""
Continuous audio buffer for real-time audio capture and processing
"""

import sounddevice as sd
import numpy as np
import threading
import time
from collections import deque
from typing import Optional, Tuple, Callable
import logging
from scipy import signal

logger = logging.getLogger(__name__)

class ContinuousAudioBuffer:
    """
    Ring buffer for continuous audio capture with thread-safe operations
    """
    
    def __init__(self, 
                 device_id: Optional[int] = None,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 buffer_size_ms: int = 20,
                 max_buffer_duration: float = 30.0):
        """
        Initialize the continuous audio buffer
        
        Args:
            device_id: Audio input device ID (-1 or None for default)
            sample_rate: Sample rate in Hz (16000 recommended for Whisper)
            channels: Number of audio channels (1 for mono)
            buffer_size_ms: Buffer chunk size in milliseconds
            max_buffer_duration: Maximum buffer duration in seconds
        """
        self.device_id = device_id if device_id != -1 else None
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size_ms = buffer_size_ms
        self.max_buffer_duration = max_buffer_duration
        
        # Calculate buffer parameters
        self.frames_per_buffer = int(sample_rate * buffer_size_ms / 1000)
        self.max_frames = int(sample_rate * max_buffer_duration)
        
        # Ring buffer for audio data
        self.audio_buffer = deque(maxlen=self.max_frames)
        self.buffer_lock = threading.Lock()
        
        # Timing and state
        self.start_time = None
        self.stream = None
        self.is_running = False
        self.total_frames_captured = 0
        
        # Statistics
        self.underruns = 0
        self.overruns = 0
        
        logger.info(f"Audio buffer initialized: {sample_rate}Hz, {channels} channel(s), "
                   f"{buffer_size_ms}ms chunks, {max_buffer_duration}s max buffer")
    
    def _audio_callback(self, indata, frames, time, status):
        """
        Audio input callback - called by sounddevice in separate thread
        
        Args:
            indata: Input audio data (numpy array)
            frames: Number of frames
            time: Timing information
            status: Status flags
        """
        if status:
            if status.input_underflow:
                self.underruns += 1
                logger.warning(f"Audio input underflow (total: {self.underruns})")
            if status.input_overflow:
                self.overruns += 1
                logger.warning(f"Audio input overflow (total: {self.overruns})")
        
        # Convert to mono if needed and flatten to 1D
        if self.channels == 1 and indata.shape[1] > 1:
            # Convert multi-channel to mono by averaging
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
        
        # Convert to int16 for compatibility with VAD
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Add to ring buffer
        with self.buffer_lock:
            self.audio_buffer.extend(audio_int16)
            self.total_frames_captured += len(audio_int16)
    
    def start(self):
        """
        Start audio capture
        """
        if self.is_running:
            logger.warning("Audio buffer already running")
            return
        
        try:
            # Query device info
            if self.device_id is not None:
                device_info = sd.query_devices(self.device_id, 'input')
                logger.info(f"Using audio device: {device_info['name']}")
            else:
                device_info = sd.query_devices(kind='input')
                logger.info(f"Using default audio device: {device_info['name']}")
            
            # Start audio stream
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=max(1, device_info['max_input_channels']) if self.device_id else 1,
                samplerate=self.sample_rate,
                blocksize=self.frames_per_buffer,
                callback=self._audio_callback,
                dtype=np.float32
            )
            
            self.stream.start()
            self.start_time = time.time()
            self.is_running = True
            
            logger.info(f"Audio capture started on device {self.device_id or 'default'}")
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            raise
    
    def stop(self):
        """
        Stop audio capture
        """
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        logger.info("Audio capture stopped")
    
    def get_frames(self, num_frames: int) -> np.ndarray:
        """
        Get specified number of frames from the buffer
        
        Args:
            num_frames: Number of frames to retrieve
            
        Returns:
            Audio data as numpy array (int16)
        """
        with self.buffer_lock:
            if len(self.audio_buffer) < num_frames:
                # Return what we have, padded with zeros if needed
                available = list(self.audio_buffer)
                if len(available) == 0:
                    return np.zeros(num_frames, dtype=np.int16)
                
                # Pad with zeros if needed
                if len(available) < num_frames:
                    padding = np.zeros(num_frames - len(available), dtype=np.int16)
                    available.extend(padding)
                
                return np.array(available[:num_frames], dtype=np.int16)
            
            # Get the last num_frames
            frames = list(self.audio_buffer)[-num_frames:]
            return np.array(frames, dtype=np.int16)
    
    def get_latest(self, duration_seconds: float) -> np.ndarray:
        """
        Get the latest audio data for specified duration
        
        Args:
            duration_seconds: Duration in seconds
            
        Returns:
            Audio data as numpy array (int16)
        """
        num_frames = int(self.sample_rate * duration_seconds)
        return self.get_frames(num_frames)
    
    def get_all_available(self) -> np.ndarray:
        """
        Get all available audio data from buffer
        
        Returns:
            Audio data as numpy array (int16)
        """
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                return np.array([], dtype=np.int16)
            return np.array(list(self.audio_buffer), dtype=np.int16)
    
    def clear_buffer(self):
        """
        Clear the audio buffer
        """
        with self.buffer_lock:
            self.audio_buffer.clear()
    
    def get_buffer_duration(self) -> float:
        """
        Get current buffer duration in seconds
        
        Returns:
            Buffer duration in seconds
        """
        with self.buffer_lock:
            return len(self.audio_buffer) / self.sample_rate if self.audio_buffer else 0.0
    
    def get_stats(self) -> dict:
        """
        Get buffer statistics
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            'buffer_duration_seconds': self.get_buffer_duration(),
            'buffer_frames': len(self.audio_buffer),
            'total_frames_captured': self.total_frames_captured,
            'underruns': self.underruns,
            'overruns': self.overruns,
            'is_running': self.is_running,
            'uptime_seconds': time.time() - self.start_time if self.start_time else 0
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
