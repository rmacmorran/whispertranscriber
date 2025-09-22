#!/usr/bin/env python3
"""
Audio device utilities for the Whisper Real-time Transcriber
"""

import sounddevice as sd
from rich.console import Console
from rich.table import Table
from typing import List, Dict, Optional

console = Console()

def list_audio_devices() -> List[Dict]:
    """
    Get list of all audio input devices with their properties
    
    Returns:
        List of device dictionaries with id, name, channels, and sample_rate info
    """
    devices = []
    device_list = sd.query_devices()
    
    for i, device in enumerate(device_list):
        if device['max_input_channels'] > 0:  # Only input devices
            devices.append({
                'id': i,
                'name': device['name'],
                'channels': device['max_input_channels'],
                'sample_rate': device['default_samplerate'],
                'hostapi': sd.query_hostapis(device['hostapi'])['name']
            })
    
    return devices

def find_device_by_name(name_fragment: str) -> Optional[int]:
    """
    Find audio device ID by name fragment (case-insensitive)
    
    Args:
        name_fragment: Part of the device name to search for
        
    Returns:
        Device ID if found, None otherwise
    """
    devices = list_audio_devices()
    name_fragment = name_fragment.lower()
    
    for device in devices:
        if name_fragment in device['name'].lower():
            return device['id']
    
    return None

def print_audio_devices():
    """
    Display available audio input devices in a formatted table
    """
    devices = list_audio_devices()
    
    if not devices:
        console.print("[red]No audio input devices found![/red]")
        return
    
    table = Table(title="Available Audio Input Devices")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Device Name", style="white")
    table.add_column("Channels", style="green", justify="center")
    table.add_column("Sample Rate", style="blue", justify="center")
    table.add_column("Host API", style="yellow")
    
    # Highlight VB-Audio devices
    for device in devices:
        name_style = "white"
        if "vb-audio" in device['name'].lower() or "virtual" in device['name'].lower():
            name_style = "bold green"
        elif "realtek" in device['name'].lower():
            name_style = "cyan"
        elif "usb" in device['name'].lower():
            name_style = "magenta"
            
        table.add_row(
            str(device['id']),
            f"[{name_style}]{device['name']}[/{name_style}]",
            str(device['channels']),
            f"{int(device['sample_rate'])} Hz",
            device['hostapi']
        )
    
    console.print(table)
    
    # Show default device
    try:
        default_device = sd.query_devices(kind='input')
        console.print(f"\n[bold]Default input device:[/bold] {default_device['name']}")
    except Exception as e:
        console.print(f"\n[red]Could not get default device: {e}[/red]")

def get_device_info(device_id: int) -> Optional[Dict]:
    """
    Get detailed information about a specific device
    
    Args:
        device_id: Device ID to query
        
    Returns:
        Device info dictionary or None if device not found
    """
    try:
        device = sd.query_devices(device_id)
        if device['max_input_channels'] == 0:
            return None
            
        return {
            'id': device_id,
            'name': device['name'],
            'channels': device['max_input_channels'],
            'sample_rate': device['default_samplerate'],
            'hostapi': sd.query_hostapis(device['hostapi'])['name'],
            'low_latency': device['default_low_input_latency'],
            'high_latency': device['default_high_input_latency']
        }
    except (ValueError, IndexError):
        return None

if __name__ == "__main__":
    print_audio_devices()
    
    # Demo: find VB-Audio Virtual Cable
    vb_audio_id = find_device_by_name("vb-audio")
    if vb_audio_id is not None:
        console.print(f"\n[green]Found VB-Audio Virtual Cable at device ID: {vb_audio_id}[/green]")
        device_info = get_device_info(vb_audio_id)
        if device_info:
            console.print(f"Latency: {device_info['low_latency']:.1f}ms - {device_info['high_latency']:.1f}ms")
    else:
        console.print(f"\n[yellow]VB-Audio Virtual Cable not found[/yellow]")
