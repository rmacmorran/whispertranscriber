#!/usr/bin/env python3
"""
Whisper transcription engine for processing audio chunks
"""

import time
import logging
import threading
import queue
import numpy as np
from typing import Optional, Dict, List, Tuple, NamedTuple
from dataclasses import dataclass
import torch
from faster_whisper import WhisperModel
from scipy.signal import resample

from vad_chunker import AudioChunk

logger = logging.getLogger(__name__)

@dataclass
class TranscriptionResult:
    """
    Result of audio transcription
    """
    chunk_id: int
    text: str
    start_time: float
    end_time: float
    confidence: float
    language: str
    processing_time: float
    segments: List[Dict]  # Detailed segment information
    
class WhisperEngine:
    """
    Whisper transcription engine with GPU acceleration and worker threads
    """
    
    def __init__(self, 
                 model_size: str = "base",
                 device: str = "auto",
                 compute_type: str = "float16",
                 num_workers: int = 2,
                 beam_size: int = 1,
                 language: Optional[str] = None,
                 word_timestamps: bool = True):
        """
        Initialize Whisper engine
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use ("cuda", "cpu", or "auto")
            compute_type: Compute type ("float16", "float32", "int8")
            num_workers: Number of worker threads for processing
            beam_size: Beam search size (1-5, higher = more accurate but slower)
            language: Language code (None for auto-detection)
            word_timestamps: Enable word-level timestamps
        """
        self.model_size = model_size
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type
        self.num_workers = num_workers
        self.beam_size = beam_size
        self.language = language
        self.word_timestamps = word_timestamps
        
        # Model and processing state
        self.model = None
        self.is_running = False
        self.workers = []
        
        # Queues for chunk processing
        self.input_queue = queue.Queue(maxsize=50)  # Pending chunks
        self.result_queue = queue.Queue()  # Completed transcriptions
        
        # Statistics
        self.chunks_processed = 0
        self.total_processing_time = 0
        self.start_time = None
        
        logger.info(f"Whisper engine initialized: {model_size} model, {self.device} device, "
                   f"{num_workers} workers, beam_size={beam_size}")
    
    def load_model(self):
        """
        Load the Whisper model
        """
        if self.model is not None:
            logger.warning("Model already loaded")
            return
        
        try:
            logger.info(f"Loading Whisper model '{self.model_size}' on {self.device}...")
            start_time = time.time()
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=None,  # Use default cache directory
                local_files_only=False
            )
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f}s")
            
            # Test transcription to warm up the model
            logger.info("Warming up model with test audio...")
            test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            _, _ = self.model.transcribe(test_audio, beam_size=1, word_timestamps=False)
            logger.info("Model warmup completed")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _worker_thread(self, worker_id: int):
        """
        Worker thread for processing audio chunks
        
        Args:
            worker_id: Unique identifier for this worker
        """
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get chunk from queue (with timeout to allow clean shutdown)
                chunk = self.input_queue.get(timeout=1.0)
                
                if chunk is None:  # Shutdown signal
                    break
                
                # Process the chunk
                result = self._transcribe_chunk(chunk, worker_id)
                
                # Put result in output queue
                self.result_queue.put(result)
                
                # Mark task as done
                self.input_queue.task_done()
                
            except queue.Empty:
                continue  # Timeout, check if still running
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                self.input_queue.task_done()
        
        logger.info(f"Worker {worker_id} stopped")
    
    def _transcribe_chunk(self, chunk: AudioChunk, worker_id: int) -> TranscriptionResult:
        """
        Transcribe a single audio chunk
        
        Args:
            chunk: Audio chunk to transcribe
            worker_id: ID of processing worker
            
        Returns:
            Transcription result
        """
        start_time = time.time()
        
        try:
            # Convert int16 audio to float32 for Whisper
            audio_float = chunk.audio_data.astype(np.float32) / 32768.0
            
            # Ensure audio is 1D
            if audio_float.ndim > 1:
                audio_float = audio_float.flatten()
            
            # Resample from 48kHz to 16kHz for Whisper
            # Whisper expects 16kHz audio, but we're capturing at 48kHz
            if chunk.sample_rate != 16000:
                target_length = int(len(audio_float) * 16000 / chunk.sample_rate)
                audio_float = resample(audio_float, target_length)
                logger.debug(f"Resampled audio from {chunk.sample_rate}Hz to 16kHz: {len(chunk.audio_data)} -> {len(audio_float)} samples")
            
            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                audio_float,
                beam_size=self.beam_size,
                language=self.language,
                word_timestamps=self.word_timestamps,
                vad_filter=False,  # We do our own VAD
                vad_parameters=None
            )
            
            # Collect segments and text
            segment_list = []
            full_text = ""
            avg_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                segment_data = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip(),
                    'avg_logprob': segment.avg_logprob,
                    'no_speech_prob': segment.no_speech_prob
                }
                
                if self.word_timestamps and hasattr(segment, 'words'):
                    segment_data['words'] = [
                        {
                            'start': word.start,
                            'end': word.end, 
                            'word': word.word,
                            'probability': word.probability
                        }
                        for word in segment.words
                    ]
                
                segment_list.append(segment_data)
                full_text += segment.text
                
                # Calculate average confidence (from log probability)
                confidence = np.exp(segment.avg_logprob) if segment.avg_logprob else 0.0
                avg_confidence += confidence
                segment_count += 1
            
            if segment_count > 0:
                avg_confidence /= segment_count
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.chunks_processed += 1
            self.total_processing_time += processing_time
            
            # Log processing info
            rtf = processing_time / chunk.duration if chunk.duration > 0 else 0
            logger.debug(f"Worker {worker_id}: Chunk {chunk.chunk_id} processed in {processing_time:.3f}s "
                        f"(RTF: {rtf:.3f}, confidence: {avg_confidence:.3f})")
            
            return TranscriptionResult(
                chunk_id=chunk.chunk_id,
                text=full_text.strip(),
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                confidence=avg_confidence,
                language=info.language,
                processing_time=processing_time,
                segments=segment_list
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Worker {worker_id}: Failed to transcribe chunk {chunk.chunk_id}: {e}")
            
            # Return empty result on error
            return TranscriptionResult(
                chunk_id=chunk.chunk_id,
                text="",
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                confidence=0.0,
                language="unknown",
                processing_time=processing_time,
                segments=[]
            )
    
    def start(self):
        """
        Start the transcription engine and worker threads
        """
        if self.is_running:
            logger.warning("Engine already running")
            return
        
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Start worker threads
        self.is_running = True
        self.start_time = time.time()
        
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_thread,
                args=(i,),
                name=f"WhisperWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Whisper engine started with {self.num_workers} workers")
    
    def stop(self):
        """
        Stop the transcription engine and worker threads
        """
        if not self.is_running:
            return
        
        logger.info("Stopping Whisper engine...")
        self.is_running = False
        
        # Send shutdown signals to workers
        for _ in self.workers:
            self.input_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not stop gracefully")
        
        self.workers.clear()
        
        # Clear queues
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
                self.input_queue.task_done()
            except queue.Empty:
                break
        
        logger.info("Whisper engine stopped")
    
    def submit_chunk(self, chunk: AudioChunk) -> bool:
        """
        Submit an audio chunk for transcription
        
        Args:
            chunk: Audio chunk to transcribe
            
        Returns:
            True if chunk was queued, False if queue is full
        """
        try:
            self.input_queue.put(chunk, block=False)
            return True
        except queue.Full:
            logger.warning(f"Input queue full, dropping chunk {chunk.chunk_id}")
            return False
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[TranscriptionResult]:
        """
        Get a transcription result
        
        Args:
            timeout: Maximum time to wait for result (None = no timeout)
            
        Returns:
            Transcription result or None if timeout/no results
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict:
        """
        Get engine statistics
        
        Returns:
            Dictionary with engine statistics
        """
        uptime = time.time() - self.start_time if self.start_time else 0
        avg_processing_time = (self.total_processing_time / self.chunks_processed 
                             if self.chunks_processed > 0 else 0)
        
        return {
            'is_running': self.is_running,
            'model_size': self.model_size,
            'device': self.device,
            'num_workers': self.num_workers,
            'chunks_processed': self.chunks_processed,
            'chunks_pending': self.input_queue.qsize(),
            'results_pending': self.result_queue.qsize(),
            'uptime_seconds': uptime,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': avg_processing_time,
            'throughput_chunks_per_second': self.chunks_processed / uptime if uptime > 0 else 0
        }
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
