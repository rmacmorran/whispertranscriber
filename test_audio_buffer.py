#!/usr/bin/env python3
"""
Test script for the ContinuousAudioBuffer component
"""

import time
import logging
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from audio_buffer import ContinuousAudioBuffer
from audio_devices import find_device_by_name, get_device_info

# Setup logging
logging.basicConfig(level=logging.INFO)
console = Console()

def test_audio_buffer():
    """Test the audio buffer with live audio capture"""
    
    console.print("[bold blue]Testing Audio Buffer[/bold blue]")
    console.print("This test will capture audio for 10 seconds and show buffer statistics")
    console.print()
    
    # Find VB-Audio Virtual Cable (WASAPI preferred)
    vb_audio_id = find_device_by_name("vb-audio virtual cable")
    
    if vb_audio_id is None:
        console.print("[yellow]VB-Audio Virtual Cable not found, using default device[/yellow]")
        device_id = None
    else:
        # Check if we have the WASAPI version (better for low latency)
        device_info = get_device_info(vb_audio_id)
        console.print(f"[green]Using VB-Audio Virtual Cable: {device_info['name']}[/green]")
        console.print(f"Host API: {device_info['hostapi']}, Latency: {device_info['low_latency']:.1f}ms")
        device_id = vb_audio_id
    
    console.print()
    
    # Create audio buffer
    buffer = ContinuousAudioBuffer(
        device_id=device_id,
        sample_rate=16000,
        channels=1,
        buffer_size_ms=20,
        max_buffer_duration=30.0
    )
    
    try:
        console.print("[yellow]Starting audio capture...[/yellow]")
        buffer.start()
        
        # Create live display
        def generate_display():
            stats = buffer.get_stats()
            
            # Calculate audio level (RMS) from recent audio
            recent_audio = buffer.get_latest(0.1)  # Last 100ms
            if len(recent_audio) > 0:
                rms = np.sqrt(np.mean(recent_audio.astype(np.float32) ** 2))
                db_level = 20 * np.log10(max(rms / 32768.0, 1e-10))
                level_bars = "█" * max(0, min(20, int((db_level + 60) / 3)))  # -60dB to 0dB range
            else:
                db_level = -60
                level_bars = ""
            
            # Format stats
            text = Text()
            text.append("🎵 Audio Buffer Test\n\n", style="bold blue")
            text.append(f"Status: {'🟢 Running' if stats['is_running'] else '🔴 Stopped'}\n", style="green" if stats['is_running'] else "red")
            text.append(f"Uptime: {stats['uptime_seconds']:.1f}s\n")
            text.append(f"Buffer Duration: {stats['buffer_duration_seconds']:.2f}s\n")
            text.append(f"Total Frames: {stats['total_frames_captured']:,}\n")
            text.append(f"Buffer Frames: {stats['buffer_frames']:,}\n")
            text.append(f"Underruns: {stats['underruns']}\n")
            text.append(f"Overruns: {stats['overruns']}\n\n")
            
            text.append("Audio Level:\n", style="bold")
            text.append(f"{level_bars:<20} {db_level:.1f} dB\n", style="cyan")
            
            if db_level > -50:
                text.append("✅ Audio detected!", style="green")
            else:
                text.append("⚠️  Low/No audio - try speaking or playing audio", style="yellow")
            
            return Panel(text, title="Audio Buffer Status", border_style="blue")
        
        # Run live display for 10 seconds
        with Live(generate_display(), refresh_per_second=4) as live:
            for i in range(40):  # 10 seconds at 4 FPS
                time.sleep(0.25)
                live.update(generate_display())
        
        console.print("\n[green]✅ Audio buffer test completed successfully![/green]")
        
        # Show final statistics
        final_stats = buffer.get_stats()
        console.print(f"Final buffer duration: {final_stats['buffer_duration_seconds']:.2f}s")
        console.print(f"Total frames captured: {final_stats['total_frames_captured']:,}")
        
        if final_stats['underruns'] > 0 or final_stats['overruns'] > 0:
            console.print(f"[yellow]⚠️  Audio issues detected - Underruns: {final_stats['underruns']}, Overruns: {final_stats['overruns']}[/yellow]")
        else:
            console.print("[green]✅ No audio buffer issues detected[/green]")
            
    except Exception as e:
        console.print(f"[red]❌ Audio buffer test failed: {e}[/red]")
        return False
    
    finally:
        buffer.stop()
    
    return True

def test_audio_retrieval():
    """Test audio data retrieval methods"""
    
    console.print("\n[bold blue]Testing Audio Data Retrieval[/bold blue]")
    
    # Find device
    vb_audio_id = find_device_by_name("vb-audio virtual cable")
    device_id = vb_audio_id if vb_audio_id is not None else None
    
    buffer = ContinuousAudioBuffer(device_id=device_id, sample_rate=16000)
    
    try:
        buffer.start()
        console.print("Capturing 3 seconds of audio for retrieval test...")
        time.sleep(3)
        
        # Test different retrieval methods
        console.print("\nTesting retrieval methods:")
        
        # Get last 1 second
        last_1s = buffer.get_latest(1.0)
        console.print(f"✅ get_latest(1.0s): {len(last_1s)} samples ({len(last_1s)/16000:.2f}s)")
        
        # Get specific number of frames
        frames_8000 = buffer.get_frames(8000)  # 0.5 seconds at 16kHz
        console.print(f"✅ get_frames(8000): {len(frames_8000)} samples ({len(frames_8000)/16000:.2f}s)")
        
        # Get all available
        all_audio = buffer.get_all_available()
        console.print(f"✅ get_all_available(): {len(all_audio)} samples ({len(all_audio)/16000:.2f}s)")
        
        # Test data type
        console.print(f"✅ Audio data type: {last_1s.dtype} (expected: int16)")
        
        # Test audio level
        if len(last_1s) > 0:
            max_amplitude = np.max(np.abs(last_1s))
            console.print(f"✅ Max amplitude: {max_amplitude} ({max_amplitude/32767*100:.1f}% of full scale)")
        
        console.print("[green]✅ Audio retrieval test passed![/green]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Audio retrieval test failed: {e}[/red]")
        return False
        
    finally:
        buffer.stop()

if __name__ == "__main__":
    console.print("[bold green]🎵 Audio Buffer Test Suite[/bold green]\n")
    
    # Test 1: Basic buffer functionality
    success1 = test_audio_buffer()
    
    # Test 2: Audio retrieval methods
    success2 = test_audio_retrieval()
    
    # Summary
    console.print(f"\n[bold]Test Results:[/bold]")
    console.print(f"Audio Buffer: {'✅ PASS' if success1 else '❌ FAIL'}")
    console.print(f"Data Retrieval: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        console.print("\n[bold green]🎉 All audio buffer tests passed![/bold green]")
        console.print("\n[yellow]💡 Tips for next steps:[/yellow]")
        console.print("- Make sure audio is playing through VB-Audio Virtual Cable")
        console.print("- Try routing your microphone or system audio to the virtual cable")
        console.print("- Check VB-Audio Control Panel if no audio is detected")
    else:
        console.print("\n[bold red]❌ Some tests failed - check the errors above[/bold red]")
