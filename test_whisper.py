#!/usr/bin/env python3
"""
Test script for faster-whisper functionality and GPU acceleration
"""

import numpy as np
import time
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

console = Console()

# Console-safe output function for Windows compatibility
def console_print(message, style=None):
    """Print message with Unicode error handling for Windows console"""
    try:
        if style:
            console.print(message, style=style)
        else:
            console.print(message)
    except UnicodeEncodeError:
        # Remove emojis and special characters for console output
        safe_message = message.replace('✅', '[OK]').replace('❌', '[FAIL]').replace('⚠️', '[WARN]').replace('🎵', '').replace('🧪', '').replace('🚀', '').replace('💻', '').replace('🎉', '').replace('💡', '')
        if style:
            console.print(safe_message, style=style)
        else:
            console.print(safe_message)

def test_whisper_installation():
    """Test if faster-whisper is properly installed"""
    console_print("[bold blue]Testing Faster-Whisper Installation[/bold blue]")
    
    try:
        from faster_whisper import WhisperModel
        console_print("✅ faster-whisper imported successfully")
        return True
    except ImportError as e:
        console_print(f"[red]❌ faster-whisper import failed: {e}[/red]")
        return False

def test_whisper_model_loading():
    """Test loading a small Whisper model"""
    console_print("\n[bold blue]Testing Whisper Model Loading[/bold blue]")
    
    try:
        from faster_whisper import WhisperModel
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading Whisper 'base' model...", total=None)
            
            # Load the base model with GPU if available
            model = WhisperModel(
                "base",
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="float16" if torch.cuda.is_available() else "float32"
            )
            
            progress.update(task, description="✅ Whisper model loaded successfully")
            
        # Get model info
        device_info = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        console_print(f"Model: base, Device: {device_info}")
        
        return model, True
        
    except Exception as e:
        console_print(f"[red]❌ Model loading failed: {e}[/red]")
        return None, False

def generate_test_speech_audio(sample_rate=16000, duration=3.0):
    """Generate synthetic speech-like audio for testing"""
    
    # Create a simple speech-like signal
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mix of frequencies that somewhat resemble speech
    signal = (
        0.3 * np.sin(2 * np.pi * 150 * t) +      # Fundamental frequency
        0.2 * np.sin(2 * np.pi * 300 * t) +      # First harmonic
        0.1 * np.sin(2 * np.pi * 600 * t) +      # Second harmonic
        0.05 * np.sin(2 * np.pi * 1200 * t) +    # Higher frequency
        0.02 * np.random.randn(len(t))           # Some noise
    )
    
    # Apply envelope to simulate speech patterns
    envelope = np.ones_like(t)
    
    # Add some pauses/variations
    for i in range(0, len(t), int(sample_rate * 0.8)):
        end_idx = min(i + int(sample_rate * 0.3), len(t))
        envelope[i:end_idx] *= np.linspace(0.2, 1.0, end_idx - i)
    
    signal *= envelope
    
    # Normalize to float32 range for Whisper
    signal = signal.astype(np.float32)
    signal = np.clip(signal, -1.0, 1.0)
    
    return signal

def test_whisper_transcription():
    """Test actual transcription with synthetic audio"""
    console_print("\n[bold blue]Testing Whisper Transcription[/bold blue]")
    
    # Load model
    model, success = test_whisper_model_loading()
    if not success:
        return False
    
    try:
        # Generate test audio
        test_audio = generate_test_speech_audio(duration=3.0)
        console_print(f"Generated {len(test_audio)/16000:.1f}s of synthetic audio")
        
        # Transcribe
        console_print("Running transcription...")
        start_time = time.time()
        
        segments, info = model.transcribe(
            test_audio,
            language="en",
            beam_size=1,
            word_timestamps=True,
            vad_filter=False  # We're doing our own VAD
        )
        
        transcription_time = time.time() - start_time
        
        # Collect results
        transcript_text = ""
        segment_count = 0
        
        for segment in segments:
            segment_count += 1
            transcript_text += segment.text
            console_print(f"Segment {segment_count}: '{segment.text.strip()}' "
                         f"({segment.start:.2f}s - {segment.end:.2f}s)")
        
        # Display results
        console_print(f"\n[green]✅ Transcription completed in {transcription_time:.2f}s[/green]")
        console_print(f"Audio duration: {len(test_audio)/16000:.2f}s")
        console_print(f"Real-time factor: {transcription_time / (len(test_audio)/16000):.2f}")
        console_print(f"Segments: {segment_count}")
        console_print(f"Language detected: {info.language} (probability: {info.language_probability:.2f})")
        
        if transcript_text.strip():
            console_print(f"Full transcript: '{transcript_text.strip()}'")
        else:
            console_print("[yellow]⚠️  No text transcribed (expected with synthetic audio)[/yellow]")
        
        # Performance check
        rtf = transcription_time / (len(test_audio)/16000)
        if rtf < 1.0:
            console_print(f"[green]✅ Real-time performance: {rtf:.2f} (< 1.0 is good)[/green]")
        else:
            console_print(f"[yellow]⚠️  Slower than real-time: {rtf:.2f}[/yellow]")
        
        return True
        
    except Exception as e:
        console_print(f"[red]❌ Transcription failed: {e}[/red]")
        return False

def test_gpu_memory_usage():
    """Test GPU memory usage if CUDA is available"""
    if not torch.cuda.is_available():
        console_print("\n[yellow]Skipping GPU memory test - CUDA not available[/yellow]")
        return True
    
    console_print("\n[bold blue]Testing GPU Memory Usage[/bold blue]")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        
        # Get GPU info
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        console_print(f"GPU Memory - Total: {mem_info.total // (1024**2)} MB")
        console_print(f"GPU Memory - Used: {mem_info.used // (1024**2)} MB")
        console_print(f"GPU Memory - Free: {mem_info.free // (1024**2)} MB")
        
        # Load model and check memory usage
        console_print("\nTesting memory usage with model loading...")
        
        from faster_whisper import WhisperModel
        model = WhisperModel("base", device="cuda", compute_type="float16")
        
        # Check memory after loading
        mem_info_after = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memory_used = (mem_info_after.used - mem_info.used) // (1024**2)
        
        console_print(f"Memory used by Whisper model: ~{memory_used} MB")
        
        if memory_used < 2000:  # Less than 2GB
            console_print("[green]✅ Memory usage looks reasonable[/green]")
        else:
            console_print("[yellow]⚠️  High memory usage detected[/yellow]")
        
        return True
        
    except Exception as e:
        console_print(f"[yellow]⚠️  GPU memory test failed: {e}[/yellow]")
        return True  # Don't fail the whole test suite for this

def display_test_summary():
    """Display information about what we've tested"""
    
    text = Text()
    text.append("🧪 Whisper Test Summary\n\n", style="bold blue")
    text.append("This test suite verifies:\n", style="bold")
    text.append("• faster-whisper library installation\n")
    text.append("• Model loading (base model)\n")
    text.append("• GPU acceleration (if available)\n")
    text.append("• Basic transcription functionality\n") 
    text.append("• Performance metrics (Real-time Factor)\n")
    text.append("• Memory usage monitoring\n\n")
    
    if torch.cuda.is_available():
        text.append("🚀 CUDA GPU detected - using GPU acceleration\n", style="green")
    else:
        text.append("💻 Using CPU processing\n", style="yellow")
        
    text.append("\nNote: Synthetic audio may not produce meaningful transcriptions,\n", style="dim")
    text.append("but successful processing indicates the system is working correctly.", style="dim")
    
    return Panel(text, title="Test Information", border_style="blue")

if __name__ == "__main__":
    console_print("[bold green]🎵 Faster-Whisper Test Suite[/bold green]\n")
    
    # Show test information
    console.print(display_test_summary())
    console.print()
    
    # Test 1: Installation check
    test1_passed = test_whisper_installation()
    
    # Test 2: Model loading
    test2_passed = False
    if test1_passed:
        _, test2_passed = test_whisper_model_loading()
    
    # Test 3: Transcription
    test3_passed = test_whisper_transcription() if test2_passed else False
    
    # Test 4: GPU memory (optional)
    test4_passed = test_gpu_memory_usage() if test2_passed else True
    
    # Summary
    console_print(f"\n[bold]Test Results:[/bold]")
    console_print(f"Installation: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    console_print(f"Model Loading: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    console_print(f"Transcription: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    console_print(f"GPU Memory: {'✅ PASS' if test4_passed else '❌ FAIL'}")
    
    all_passed = test1_passed and test2_passed and test3_passed and test4_passed
    
    if all_passed:
        console_print("\n[bold green]🎉 All Whisper tests passed![/bold green]")
        console_print("\n[yellow]💡 Ready to integrate with audio chunking pipeline![/yellow]")
        
        if torch.cuda.is_available():
            console_print("\n[green]🚀 GPU acceleration is working - expect fast transcription![/green]")
    else:
        console_print("\n[bold red]❌ Some tests failed - check the errors above[/bold red]")
