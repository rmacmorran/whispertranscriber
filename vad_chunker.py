#!/usr/bin/env python3
"""
Voice Activity Detection based audio chunker with intelligent silence detection
"""

import numpy as np
import threading
import time
import logging
from collections import deque
from typing import Optional, List, Tuple, NamedTuple
from dataclasses import dataclass
import torch
from scipy import signal

# Import Silero VAD
from silero_vad import load_silero_vad, get_speech_timestamps

logger = logging.getLogger(__name__)

@dataclass
class AudioChunk:
    """
    Audio chunk with metadata
    """
    audio_data: np.ndarray  # int16 audio data
    start_time: float       # Start time in seconds since stream start
    end_time: float         # End time in seconds since stream start
    duration: float         # Duration in seconds
    contains_speech: bool   # Whether chunk contains speech
    chunk_id: int          # Unique chunk identifier
    sample_rate: int       # Sample rate of the audio data

class VADChunker:
    """
    Voice Activity Detection based audio chunker that creates chunks at silence boundaries
    """
    
    def __init__(self,
                 sample_rate: int = 16000,
                 min_silence_ms: int = 300,
                 min_chunk_duration: float = 2.0,
                 max_chunk_duration: float = 12.0,
                 chunk_overlap: float = 0.5,
                 vad_threshold: float = 0.5):
        """
        Initialize VAD chunker
        
        Args:
            sample_rate: Audio sample rate in Hz
            min_silence_ms: Minimum silence duration to trigger chunk boundary (ms)
            min_chunk_duration: Minimum chunk duration in seconds
            max_chunk_duration: Maximum chunk duration in seconds  
            chunk_overlap: Overlap between chunks in seconds
            vad_threshold: VAD sensitivity (0.1 to 1.0, higher = more sensitive)
        """
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_chunk_duration = min_chunk_duration
        self.max_chunk_duration = max_chunk_duration
        self.chunk_overlap = chunk_overlap
        self.vad_threshold = vad_threshold
        
        # Initialize Silero VAD model
        try:
            self.vad_model = load_silero_vad(onnx=True)  # Use ONNX for better performance
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise
        
        # Chunking state
        self.current_chunk_audio = deque()
        self.current_chunk_start_time = None
        self.chunk_counter = 0
        self.last_chunk_end_audio = deque(maxlen=int(sample_rate * chunk_overlap))
        
        # Timing
        self.stream_start_time = time.time()
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info(f"VAD Chunker initialized: {min_silence_ms}ms silence, "
                   f"{min_chunk_duration}-{max_chunk_duration}s chunks, "
                   f"{chunk_overlap}s overlap, threshold={vad_threshold}")
    
    def detect_speech_segments(self, audio_data: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect speech segments in audio data using Silero VAD
        
        Args:
            audio_data: Audio data as int16 numpy array
            
        Returns:
            List of (start_sample, end_sample) tuples for speech segments
        """
        # Convert int16 to float32 for Silero VAD
        audio_float = audio_data.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float)
        
        try:
            # Get speech timestamps (returns list of dicts with 'start' and 'end' sample indices)
            speech_timestamps = get_speech_timestamps(
                audio_tensor,
                self.vad_model,
                sampling_rate=self.sample_rate,
                threshold=self.vad_threshold,
                min_speech_duration_ms=100,  # Minimum speech segment duration
                min_silence_duration_ms=self.min_silence_ms
            )
            
            # Convert to list of tuples
            segments = [(ts['start'], ts['end']) for ts in speech_timestamps]
            return segments
            
        except Exception as e:
            logger.warning(f"VAD processing failed: {e}")
            # Fallback: assume entire audio is speech
            return [(0, len(audio_data))]
    
    def find_silence_boundary(self, audio_data: np.ndarray, start_idx: int = 0) -> Optional[int]:
        """
        Find the next silence boundary in audio data
        
        Args:
            audio_data: Audio data as int16 numpy array
            start_idx: Starting index to search from
            
        Returns:
            Sample index of silence boundary, or None if no silence found
        """
        if len(audio_data) <= start_idx:
            return None
        
        search_audio = audio_data[start_idx:]
        speech_segments = self.detect_speech_segments(search_audio)
        
        if not speech_segments:
            # No speech detected, entire segment is silence
            return start_idx
        
        # Find gap between speech segments that's long enough
        min_silence_samples = int(self.sample_rate * self.min_silence_ms / 1000)
        
        for i in range(len(speech_segments) - 1):
            current_end = speech_segments[i][1]
            next_start = speech_segments[i + 1][0]
            
            silence_duration = next_start - current_end
            if silence_duration >= min_silence_samples:
                # Found suitable silence gap
                silence_center = current_end + silence_duration // 2
                return start_idx + silence_center
        
        # Check silence at the end
        last_segment_end = speech_segments[-1][1]
        if len(search_audio) - last_segment_end >= min_silence_samples:
            return start_idx + last_segment_end + min_silence_samples // 2
        
        return None
    
    def should_force_chunk(self, current_duration: float) -> bool:
        """
        Check if we should force a chunk boundary due to maximum duration
        
        Args:
            current_duration: Current chunk duration in seconds
            
        Returns:
            True if chunk should be forced
        """
        return current_duration >= self.max_chunk_duration
    
    def create_chunk(self, audio_data: np.ndarray, start_time: float, end_time: float) -> AudioChunk:
        """
        Create an audio chunk with metadata
        
        Args:
            audio_data: Audio data as int16 numpy array
            start_time: Start time relative to stream start
            end_time: End time relative to stream start
            
        Returns:
            AudioChunk object
        """
        duration = end_time - start_time
        
        # Check if chunk contains speech
        speech_segments = self.detect_speech_segments(audio_data)
        contains_speech = len(speech_segments) > 0
        
        chunk = AudioChunk(
            audio_data=audio_data.copy(),
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            contains_speech=contains_speech,
            chunk_id=self.chunk_counter,
            sample_rate=self.sample_rate
        )
        
        self.chunk_counter += 1
        return chunk
    
    def add_overlap_to_chunk(self, chunk_audio: np.ndarray) -> np.ndarray:
        """
        Add overlap from previous chunk to current chunk
        
        Args:
            chunk_audio: Current chunk audio data
            
        Returns:
            Chunk audio with overlap prepended
        """
        if not self.last_chunk_end_audio:
            return chunk_audio
        
        overlap_audio = np.array(list(self.last_chunk_end_audio), dtype=np.int16)
        return np.concatenate([overlap_audio, chunk_audio])
    
    def update_overlap_buffer(self, chunk_audio: np.ndarray):
        """
        Update the overlap buffer with the end of current chunk
        
        Args:
            chunk_audio: Current chunk audio data
        """
        overlap_samples = int(self.sample_rate * self.chunk_overlap)
        if len(chunk_audio) >= overlap_samples:
            # Store the last overlap_samples for next chunk
            self.last_chunk_end_audio.clear()
            self.last_chunk_end_audio.extend(chunk_audio[-overlap_samples:])
    
    def process_audio(self, audio_data: np.ndarray, current_time: float) -> List[AudioChunk]:
        """
        Process incoming audio data and return completed chunks
        
        Args:
            audio_data: New audio data as int16 numpy array
            current_time: Current time in seconds since stream start
            
        Returns:
            List of completed audio chunks
        """
        chunks = []
        
        with self.lock:
            # Add new audio to current chunk
            self.current_chunk_audio.extend(audio_data)
            
            # Initialize start time if this is the first chunk
            if self.current_chunk_start_time is None:
                self.current_chunk_start_time = current_time - len(audio_data) / self.sample_rate
            
            # Convert current chunk to numpy array
            current_chunk_array = np.array(list(self.current_chunk_audio), dtype=np.int16)
            current_duration = len(current_chunk_array) / self.sample_rate
            
            # Check if we should create a chunk
            should_chunk = False
            chunk_end_idx = len(current_chunk_array)
            
            # Check for maximum duration limit
            if self.should_force_chunk(current_duration):
                should_chunk = True
                # Try to find a silence boundary to cut at
                search_start = max(0, int(self.sample_rate * self.min_chunk_duration))
                silence_idx = self.find_silence_boundary(current_chunk_array, search_start)
                if silence_idx is not None:
                    chunk_end_idx = silence_idx
                logger.debug(f"Forced chunk due to max duration ({current_duration:.1f}s)")
            
            # Check for minimum duration + silence boundary
            elif current_duration >= self.min_chunk_duration:
                search_start = int(self.sample_rate * self.min_chunk_duration)
                silence_idx = self.find_silence_boundary(current_chunk_array, search_start)
                if silence_idx is not None:
                    should_chunk = True
                    chunk_end_idx = silence_idx
                    logger.debug(f"Natural chunk at silence boundary ({current_duration:.1f}s)")
            
            # Create chunk if needed
            if should_chunk and len(current_chunk_array) > 0:
                # Extract chunk audio
                chunk_audio = current_chunk_array[:chunk_end_idx]
                
                # Add overlap from previous chunk
                chunk_with_overlap = self.add_overlap_to_chunk(chunk_audio)
                
                # Calculate timing
                chunk_start = self.current_chunk_start_time
                chunk_end = chunk_start + len(chunk_audio) / self.sample_rate
                
                # Create chunk
                chunk = self.create_chunk(chunk_with_overlap, chunk_start, chunk_end)
                chunks.append(chunk)
                
                logger.info(f"Created chunk {chunk.chunk_id}: {chunk.duration:.2f}s "
                           f"({len(chunk.audio_data)} samples, speech: {chunk.contains_speech})")
                
                # Update overlap buffer
                self.update_overlap_buffer(chunk_audio)
                
                # Keep remaining audio for next chunk
                remaining_audio = current_chunk_array[chunk_end_idx:]
                self.current_chunk_audio.clear()
                self.current_chunk_audio.extend(remaining_audio)
                
                # Update start time for next chunk
                self.current_chunk_start_time = chunk_end
        
        return chunks
    
    def finalize_current_chunk(self) -> Optional[AudioChunk]:
        """
        Finalize the current chunk (called when stopping)
        
        Returns:
            Final chunk if available, None otherwise
        """
        with self.lock:
            if not self.current_chunk_audio:
                return None
            
            current_chunk_array = np.array(list(self.current_chunk_audio), dtype=np.int16)
            current_duration = len(current_chunk_array) / self.sample_rate
            
            # Only create chunk if it meets minimum duration or contains speech
            if current_duration < 1.0:  # Very short chunks are probably noise
                speech_segments = self.detect_speech_segments(current_chunk_array)
                if not speech_segments:
                    return None
            
            # Add overlap from previous chunk
            chunk_with_overlap = self.add_overlap_to_chunk(current_chunk_array)
            
            # Calculate timing
            chunk_start = self.current_chunk_start_time or 0
            chunk_end = chunk_start + current_duration
            
            # Create final chunk
            chunk = self.create_chunk(chunk_with_overlap, chunk_start, chunk_end)
            
            logger.info(f"Finalized chunk {chunk.chunk_id}: {chunk.duration:.2f}s "
                       f"({len(chunk.audio_data)} samples, speech: {chunk.contains_speech})")
            
            # Clear state
            self.current_chunk_audio.clear()
            self.current_chunk_start_time = None
            
            return chunk
    
    def reset(self):
        """
        Reset chunker state
        """
        with self.lock:
            self.current_chunk_audio.clear()
            self.current_chunk_start_time = None
            self.last_chunk_end_audio.clear()
            self.chunk_counter = 0
            self.stream_start_time = time.time()
            
        logger.info("VAD chunker reset")
