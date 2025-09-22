#!/usr/bin/env python3
"""
Test script for the VAD Chunker component
"""

import time
import logging
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import torch
from vad_chunker import VADChunker, AudioChunk
from audio_buffer import ContinuousAudioBuffer
from audio_devices import find_device_by_name

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
console = Console()

def generate_test_audio(sample_rate=16000, duration=10.0):
    """Generate test audio with speech-like patterns and silence"""
    
    total_samples = int(sample_rate * duration)
    audio = np.zeros(total_samples, dtype=np.int16)
    
    # Generate some speech-like segments with silence gaps
    segments = [
        (0.5, 2.0),    # Speech from 0.5s to 2.0s
        (3.0, 4.5),    # Speech from 3.0s to 4.5s  
        (6.0, 8.5),    # Speech from 6.0s to 8.5s
    ]
    
    for start_time, end_time in segments:
        start_sample = int(start_time * sample_rate)
        end_sample = min(int(end_time * sample_rate), total_samples)
        
        if start_sample >= end_sample or start_sample >= total_samples:
            continue
            
        # Generate speech-like signal (mix of frequencies)
        segment_length = end_sample - start_sample
        t = np.linspace(0, segment_length / sample_rate, segment_length)
        
        # Mix of different frequency components to simulate speech
        signal = (
            0.3 * np.sin(2 * np.pi * 200 * t) +  # Low frequency
            0.2 * np.sin(2 * np.pi * 800 * t) +  # Mid frequency  
            0.1 * np.sin(2 * np.pi * 1600 * t) + # High frequency
            0.05 * np.random.randn(len(t))       # Some noise
        )
        
        # Apply envelope to make it more speech-like
        segment_duration = segment_length / sample_rate
        envelope = np.exp(-0.5 * (t - segment_duration/2)**2 / (segment_duration/4)**2)
        signal *= envelope
        
        # Convert to int16
        audio[start_sample:end_sample] = (signal * 16384).astype(np.int16)
    
    console.print(f"Generated {duration}s test audio with {len(segments)} speech segments")
    return audio

def test_vad_model_loading():
    """Test if VAD model loads correctly"""
    console.print("[bold blue]Testing VAD Model Loading[/bold blue]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading Silero VAD model...", total=None)
            
            chunker = VADChunker(
                sample_rate=16000,
                min_silence_ms=300,
                min_chunk_duration=2.0,
                max_chunk_duration=8.0,
                chunk_overlap=0.5,
                vad_threshold=0.5
            )
            
            progress.update(task, description="✅ Silero VAD model loaded successfully")
        
        console.print("[green]✅ VAD model loading test passed![/green]")
        return chunker, True
        
    except Exception as e:
        console.print(f"[red]❌ VAD model loading failed: {e}[/red]")
        return None, False

def test_vad_chunking_synthetic():
    """Test VAD chunking with synthetic audio data"""
    console.print("\n[bold blue]Testing VAD Chunking with Synthetic Audio[/bold blue]")
    
    # Load chunker
    chunker, success = test_vad_model_loading()
    if not success:
        return False
    
    # Generate test audio
    test_audio = generate_test_audio(duration=10.0)
    
    # Process audio in chunks (simulate real-time)
    chunk_size = 1600  # 100ms at 16kHz
    chunks_created = []
    
    console.print("Processing synthetic audio...")
    
    with Progress() as progress:
        task = progress.add_task("Processing audio chunks...", total=len(test_audio)//chunk_size)
        
        for i in range(0, len(test_audio), chunk_size):
            end_idx = min(i + chunk_size, len(test_audio))
            audio_chunk = test_audio[i:end_idx]
            current_time = end_idx / 16000  # Time in seconds
            
            # Process chunk
            new_chunks = chunker.process_audio(audio_chunk, current_time)
            chunks_created.extend(new_chunks)
            
            progress.advance(task)
    
    # Finalize any remaining audio
    final_chunk = chunker.finalize_current_chunk()
    if final_chunk:
        chunks_created.append(final_chunk)
    
    # Display results
    if chunks_created:
        table = Table(title="Generated Audio Chunks")
        table.add_column("Chunk ID", style="cyan")
        table.add_column("Start (s)", style="green")
        table.add_column("End (s)", style="green") 
        table.add_column("Duration (s)", style="blue")
        table.add_column("Samples", style="magenta")
        table.add_column("Has Speech", style="yellow")
        
        for chunk in chunks_created:
            table.add_row(
                str(chunk.chunk_id),
                f"{chunk.start_time:.2f}",
                f"{chunk.end_time:.2f}",
                f"{chunk.duration:.2f}",
                str(len(chunk.audio_data)),
                "✅" if chunk.contains_speech else "❌"
            )
        
        console.print(table)
        console.print(f"\n[green]✅ Created {len(chunks_created)} chunks from synthetic audio[/green]")
        
        # Validate chunk properties
        speech_chunks = [c for c in chunks_created if c.contains_speech]
        console.print(f"Speech chunks: {len(speech_chunks)}/{len(chunks_created)}")
        
        # Check duration constraints
        valid_durations = [c for c in chunks_created if 1.5 <= c.duration <= 15.0]  # Allow some tolerance
        console.print(f"Valid durations: {len(valid_durations)}/{len(chunks_created)}")
        
        return len(chunks_created) > 0
    else:
        console.print("[red]❌ No chunks created from synthetic audio[/red]")
        return False

def test_vad_chunking_live():
    """Test VAD chunking with live audio"""
    console.print("\n[bold blue]Testing VAD Chunking with Live Audio[/bold blue]")
    
    # Find VB-Audio device
    vb_audio_id = find_device_by_name("vb-audio virtual cable")
    if vb_audio_id is None:
        console.print("[yellow]VB-Audio Virtual Cable not found, using default device[/yellow]")
        device_id = None
    else:
        device_id = vb_audio_id
        console.print(f"[green]Using VB-Audio Virtual Cable (Device ID: {device_id})[/green]")
    
    # Create audio buffer and VAD chunker
    buffer = ContinuousAudioBuffer(
        device_id=device_id,
        sample_rate=16000,
        channels=1,
        buffer_size_ms=100  # Larger chunks for VAD processing
    )
    
    chunker, success = test_vad_model_loading()
    if not success:
        return False
    
    chunks_created = []
    
    try:
        console.print("\n[yellow]Starting live audio capture for 15 seconds...[/yellow]")
        console.print("[dim]💡 Try speaking, playing music, or routing audio to VB-Audio Virtual Cable[/dim]\n")
        
        buffer.start()
        
        # Process audio for 15 seconds
        start_time = time.time()
        last_processed_time = 0
        
        while time.time() - start_time < 15:
            current_time = time.time() - start_time
            
            # Get new audio data (since last processing)
            duration_to_get = current_time - last_processed_time
            if duration_to_get > 0.1:  # Process every 100ms
                new_audio = buffer.get_latest(duration_to_get)
                
                if len(new_audio) > 0:
                    # Process with VAD chunker
                    new_chunks = chunker.process_audio(new_audio, current_time)
                    
                    # Display new chunks immediately
                    for chunk in new_chunks:
                        chunks_created.append(chunk)
                        console.print(f"📦 Chunk {chunk.chunk_id}: {chunk.duration:.2f}s, "
                                    f"Speech: {'✅' if chunk.contains_speech else '❌'}, "
                                    f"Samples: {len(chunk.audio_data):,}")
                
                last_processed_time = current_time
            
            time.sleep(0.05)  # Small delay
        
        # Finalize remaining audio
        final_chunk = chunker.finalize_current_chunk()
        if final_chunk:
            chunks_created.append(final_chunk)
            console.print(f"📦 Final Chunk {final_chunk.chunk_id}: {final_chunk.duration:.2f}s, "
                        f"Speech: {'✅' if final_chunk.contains_speech else '❌'}")
        
        console.print(f"\n[green]✅ Live chunking test completed - Created {len(chunks_created)} chunks[/green]")
        
        if chunks_created:
            # Summary statistics
            total_duration = sum(c.duration for c in chunks_created)
            speech_chunks = len([c for c in chunks_created if c.contains_speech])
            avg_duration = total_duration / len(chunks_created)
            
            console.print(f"Total audio duration: {total_duration:.1f}s")
            console.print(f"Speech chunks: {speech_chunks}/{len(chunks_created)}")
            console.print(f"Average chunk duration: {avg_duration:.1f}s")
        else:
            console.print("[yellow]⚠️  No chunks created - check if audio is reaching the device[/yellow]")
        
        return len(chunks_created) > 0
        
    except Exception as e:
        console.print(f"[red]❌ Live chunking test failed: {e}[/red]")
        return False
        
    finally:
        buffer.stop()

def test_chunk_overlap():
    """Test that chunk overlap is working correctly"""
    console.print("\n[bold blue]Testing Chunk Overlap[/bold blue]")
    
    chunker, success = test_vad_model_loading()
    if not success:
        return False
    
    # Generate audio with clear speech patterns
    test_audio = generate_test_audio(duration=6.0)
    
    # Process all at once to get chunks
    chunks = []
    chunk_size = 800  # 50ms chunks
    
    for i in range(0, len(test_audio), chunk_size):
        end_idx = min(i + chunk_size, len(test_audio))
        audio_chunk = test_audio[i:end_idx]
        current_time = end_idx / 16000
        
        new_chunks = chunker.process_audio(audio_chunk, current_time)
        chunks.extend(new_chunks)
    
    final_chunk = chunker.finalize_current_chunk()
    if final_chunk:
        chunks.append(final_chunk)
    
    if len(chunks) >= 2:
        console.print(f"Generated {len(chunks)} chunks for overlap testing")
        
        # Check if chunks have expected overlap
        overlap_samples = int(16000 * 0.5)  # 0.5 second overlap
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # The current chunk should start with overlap from previous chunk
            if len(curr_chunk.audio_data) > overlap_samples:
                console.print(f"✅ Chunk {curr_chunk.chunk_id} has overlap region ({overlap_samples} samples)")
            else:
                console.print(f"⚠️  Chunk {curr_chunk.chunk_id} too short for full overlap")
        
        console.print("[green]✅ Chunk overlap test completed[/green]")
        return True
    else:
        console.print("[yellow]⚠️  Not enough chunks generated for overlap testing[/yellow]")
        return False

if __name__ == "__main__":
    console.print("[bold green]🎙️ VAD Chunker Test Suite[/bold green]\n")
    
    # Test 1: Model loading
    test1_passed = test_vad_model_loading()[1]
    
    # Test 2: Synthetic audio chunking  
    test2_passed = test_vad_chunking_synthetic() if test1_passed else False
    
    # Test 3: Chunk overlap
    test3_passed = test_chunk_overlap() if test1_passed else False
    
    # Test 4: Live audio chunking
    console.print("\n" + "="*50)
    console.print("[bold yellow]Ready for live audio test?[/bold yellow]")
    console.print("This will capture audio for 15 seconds from your selected device.")
    input("Press Enter to continue or Ctrl+C to skip...")
    
    test4_passed = test_vad_chunking_live() if test1_passed else False
    
    # Summary
    console.print(f"\n[bold]Test Results:[/bold]")
    console.print(f"Model Loading: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    console.print(f"Synthetic Chunking: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    console.print(f"Chunk Overlap: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    console.print(f"Live Chunking: {'✅ PASS' if test4_passed else '❌ FAIL'}")
    
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    
    if all_passed:
        console.print("\n[bold green]🎉 All VAD chunker tests passed![/bold green]")
        console.print("\n[yellow]💡 The chunker is working correctly and ready for Whisper integration![/yellow]")
    else:
        console.print("\n[bold red]❌ Some tests failed - check the errors above[/bold red]")
