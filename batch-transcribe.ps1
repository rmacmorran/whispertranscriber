# Batch Transcribe Script
# Processes all video and audio files in the current directory with Whisper Transcriber

param(
    [string]$OutputDir = ".",
    [switch]$Timestamps,
    [string]$Model = "base",
    [switch]$Force,
    [switch]$ContinueOnError,
    [switch]$CpuOnly,
    [switch]$Subtitle,
    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host @"
Batch Transcribe Script - Process all video/audio files with Whisper Transcriber

USAGE:
    .\batch-transcribe.ps1 [OPTIONS]

OPTIONS:
    -OutputDir <path>   Output directory for transcripts (default: current directory)
    -Timestamps         Include timestamps in transcripts
    -Model <size>       Whisper model size (tiny/base/small/medium/large/large-v2/large-v3, default: 'base')
    -Force              Overwrite existing transcript files
    -ContinueOnError    Continue processing other files if one fails
    -CpuOnly            Force CPU-only processing (avoids GPU-related crashes)
    -Subtitle           Generate subtitle files (.srt) instead of text files (.txt)
    -Help               Show this help message

EXAMPLES:
    .\batch-transcribe.ps1                                    # Basic transcription
    .\batch-transcribe.ps1 -Timestamps                       # With timestamps
    .\batch-transcribe.ps1 -OutputDir "my_transcripts"       # Custom output folder
    .\batch-transcribe.ps1 -Model small -Timestamps -Force   # Advanced options
    .\batch-transcribe.ps1 -CpuOnly                          # Force CPU processing (if GPU crashes)
    .\batch-transcribe.ps1 -Subtitle                         # Generate SRT subtitle files
    .\batch-transcribe.ps1 -Subtitle -Model medium           # Generate subtitles with better model

SUPPORTED FORMATS:
    Video: .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm, .m4v
    Audio: .mp3, .wav, .flac, .aac, .ogg, .m4a, .wma
"@
    exit 0
}

# Validate command-line parameters
# Valid Whisper model sizes
$validModels = @('tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3')

# Function to validate parameters
function Test-Parameters {
    $errors = @()
    
    # Validate Model parameter
    if ($Model -and $Model -notin $validModels) {
        $errors += "Invalid model size: '$Model'. Valid options are: $($validModels -join ', ')"
    }
    
    # Validate OutputDir parameter
    if ($OutputDir) {
        try {
            # Test if we can resolve/create the path
            $resolvedPath = $null
            if (Test-Path $OutputDir) {
                $resolvedPath = Resolve-Path $OutputDir -ErrorAction Stop
            } else {
                # Try to create parent directories to validate path
                $parentPath = Split-Path $OutputDir -Parent
                if ($parentPath -and -not (Test-Path $parentPath)) {
                    $errors += "Output directory parent path does not exist: '$parentPath'"
                }
            }
        }
        catch {
            $errors += "Invalid output directory path: '$OutputDir' - $($_.Exception.Message)"
        }
    }
    
    return $errors
}

# Validate parameters before proceeding
Write-Host "Validating command-line parameters..." -ForegroundColor Cyan
$validationErrors = Test-Parameters

if ($validationErrors.Count -gt 0) {
    Write-Host "Parameter validation failed:" -ForegroundColor Red
    foreach ($error in $validationErrors) {
        Write-Host "  [FAIL] $error" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Use -Help to see valid options and examples." -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Parameters validated successfully" -ForegroundColor Green
Write-Host ""

# Supported file extensions
$videoExtensions = @('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v')
$audioExtensions = @('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma')
$allExtensions = $videoExtensions + $audioExtensions

# Colors for output
$colors = @{
    Info = 'Cyan'
    Success = 'Green' 
    Warning = 'Yellow'
    Error = 'Red'
    Progress = 'Magenta'
}

# Function to write colored output
function Write-ColorOutput {
    param([string]$Message, [string]$Color = 'White')
    Write-Host $Message -ForegroundColor $colors[$Color]
}

# Function to get file size in human readable format
function Get-FileSize {
    param([long]$Bytes)
    if ($Bytes -ge 1GB) { return "{0:F2} GB" -f ($Bytes / 1GB) }
    elseif ($Bytes -ge 1MB) { return "{0:F2} MB" -f ($Bytes / 1MB) }
    elseif ($Bytes -ge 1KB) { return "{0:F2} KB" -f ($Bytes / 1KB) }
    else { return "$Bytes bytes" }
}

# Function to format duration
function Format-Duration {
    param([TimeSpan]$Duration)
    if ($Duration.TotalHours -ge 1) {
        return "{0:hh\:mm\:ss}" -f $Duration
    } else {
        return "{0:mm\:ss}" -f $Duration
    }
}

# Find whisper-transcriber.py - check multiple locations
$scriptPath = $null
$configPath = $null

# Try environment variable first (if set)
if ($env:WHISPER_TRANSCRIBER_PATH -and (Test-Path $env:WHISPER_TRANSCRIBER_PATH)) {
    $scriptPath = $env:WHISPER_TRANSCRIBER_PATH
    $configPath = Join-Path (Split-Path $scriptPath -Parent) "config.yaml"
}
# Try current directory
elseif (Test-Path ".\whisper-transcriber.py") {
    $scriptPath = Resolve-Path ".\whisper-transcriber.py"
    $configPath = Resolve-Path ".\config.yaml" -ErrorAction SilentlyContinue
}
# Try the directory where this batch script is located
elseif (Test-Path (Join-Path $PSScriptRoot "whisper-transcriber.py")) {
    $scriptPath = Join-Path $PSScriptRoot "whisper-transcriber.py"
    $configPath = Join-Path $PSScriptRoot "config.yaml"
}
# Try common installation paths
elseif (Test-Path "C:\Users\rmacmorran\projects\whisper-transcriber\whisper-transcriber.py") {
    $scriptPath = "C:\Users\rmacmorran\projects\whisper-transcriber\whisper-transcriber.py"
    $configPath = "C:\Users\rmacmorran\projects\whisper-transcriber\config.yaml"
}

if (-not $scriptPath -or -not (Test-Path $scriptPath)) {
    Write-ColorOutput "whisper-transcriber.py not found!" 'Error'
    Write-ColorOutput "Searched in:" 'Warning'
    Write-ColorOutput "  - Current directory: $(Get-Location)" 'Info'
    Write-ColorOutput "  - Script directory: $PSScriptRoot" 'Info'
    Write-ColorOutput "  - Default install: C:\Users\rmacmorran\projects\whisper-transcriber\" 'Info'
    Write-ColorOutput "" 'Info'
    Write-ColorOutput "Solutions:" 'Info'
    Write-ColorOutput "  1. Run from the whisper-transcriber project folder, or" 'Info'
    Write-ColorOutput "  2. Copy this script to your whisper-transcriber project folder, or" 'Info'
    Write-ColorOutput "  3. Set the WHISPER_TRANSCRIBER_PATH environment variable" 'Info'
    exit 1
}

Write-ColorOutput "Using transcriber script: $scriptPath" 'Info'

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Write-ColorOutput "Created output directory: $OutputDir" 'Info'
}

# Find all video/audio files
Write-ColorOutput "Scanning for video and audio files..." 'Info'
$allFiles = Get-ChildItem -Path "." -File | Where-Object { $_.Extension.ToLower() -in $allExtensions }

if ($allFiles.Count -eq 0) {
    Write-ColorOutput "No video or audio files found in current directory!" 'Warning'
    Write-ColorOutput "Supported formats: $($allExtensions -join ', ')" 'Info'
    exit 0
}

Write-ColorOutput "Found $($allFiles.Count) files to process:" 'Success'
$allFiles | ForEach-Object {
    $size = Get-FileSize $_.Length
    Write-ColorOutput "   $($_.Name) ($size)" 'Info'
}
Write-Host ""

# Test Python environment and dependencies before processing
Write-ColorOutput "Testing Python environment..." 'Info'
try {
    $testResult = & python -c @"
try:
    import faster_whisper
    import librosa
    import soundfile
    import numpy as np
    print('DEPS_OK')
except Exception as e:
    print(f'DEPS_ERROR: {e}')
"@ 2>&1
    
    if ($testResult -contains 'DEPS_OK') {
        Write-ColorOutput "[OK] Python dependencies validated" 'Success'
    } else {
        Write-ColorOutput "[FAIL] Python dependency issues detected:" 'Error'
        foreach ($line in $testResult) {
            Write-ColorOutput "   $line" 'Error'
        }
        Write-ColorOutput "Please ensure all required packages are installed" 'Warning'
        exit 1
    }
} catch {
    Write-ColorOutput "[FAIL] Failed to test Python environment: $($_.Exception.Message)" 'Error'
    exit 1
}

# Build base command arguments
$baseArgs = @('-q')  # Quiet mode
if ($Timestamps) { $baseArgs += '-t' }
if ($Model -ne 'base') { $baseArgs += @('-m', $Model) }

# Add config file path with smart CPU detection
if ($CpuOnly -and (Test-Path 'config-cpu.yaml')) {
    $baseArgs += @('-c', 'config-cpu.yaml')
    Write-ColorOutput "Using CPU-only config file: config-cpu.yaml" 'Warning'
} elseif ($configPath -and (Test-Path $configPath)) {
    # Check if we need to override config for CPU processing
    try {
        $gpuAvailable = & python -c "import torch; print('GPU' if torch.cuda.is_available() else 'CPU')" 2>$null
        if ($gpuAvailable -eq 'CPU') {
            Write-ColorOutput "GPU not available - consider using -CpuOnly flag for optimal performance" 'Warning'
        }
    } catch {
        # Ignore GPU detection errors
    }
    
    $baseArgs += @('-c', $configPath)
    Write-ColorOutput "Using config file: $configPath" 'Info'
} else {
    Write-ColorOutput "Config file not found, using defaults" 'Warning'
}

# Process each file
$successful = 0
$failed = 0
$skipped = 0
$startTime = Get-Date

Write-ColorOutput "Starting batch transcription..." 'Progress'
Write-ColorOutput "Model: $Model" 'Info'
Write-ColorOutput "Timestamps: $(if ($Timestamps) { 'Yes' } else { 'No' })" 'Info'
Write-ColorOutput "Format: $(if ($Subtitle) { 'SRT Subtitles' } else { 'Text Transcripts' })" 'Info'
Write-ColorOutput "Output: $(if ($OutputDir -eq '.') { 'current directory' } else { "$OutputDir\" })" 'Info'
Write-Host ""

for ($i = 0; $i -lt $allFiles.Count; $i++) {
    $file = $allFiles[$i]
    $progress = [math]::Round(($i / $allFiles.Count) * 100, 1)
    
    # Generate output filename
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    $outputExtension = if ($Subtitle) { ".srt" } else { ".txt" }
    $outputFile = Join-Path $OutputDir "$baseName$outputExtension"
    
    Write-ColorOutput "[$($i + 1)/$($allFiles.Count)] ($progress%) Processing: $($file.Name)" 'Progress'
    
    # Check if output already exists
    if ((Test-Path $outputFile) -and -not $Force) {
        Write-ColorOutput "   Transcript already exists, skipping (use -Force to overwrite)" 'Warning'
        $skipped++
        Write-Host ""
        continue
    }
    
    # Build command arguments for this file
    $fileArgs = $baseArgs + @('-i', $file.FullName, '-o', $outputFile)
    
    try {
        $fileStartTime = Get-Date
        
        # Debug: Show the exact command being run
        $commandString = "python `"$scriptPath`" " + ($fileArgs -join ' ')
        Write-ColorOutput "   Command: $commandString" 'Info'
        
        # Run transcription
        Write-ColorOutput "   Transcribing..." 'Info'
        $result = & python $scriptPath @fileArgs 2>&1
        
        $fileEndTime = Get-Date
        $processingTime = $fileEndTime - $fileStartTime
        
        # More detailed exit code analysis
        if ($LASTEXITCODE -eq 0) {
            if (Test-Path $outputFile) {
                $transcriptSize = Get-FileSize (Get-Item $outputFile).Length
                Write-ColorOutput "   Success! Transcript saved ($transcriptSize) - took $(Format-Duration $processingTime)" 'Success'
                $successful++
            } else {
                Write-ColorOutput "   Error: Transcript file was not created" 'Error'
                $failed++
            }
        } else {
            # Decode Windows error codes (handle negative values)
            # Convert negative exit code to hex without uint32 casting
            if ($LASTEXITCODE -lt 0) {
                $hexCode = "0x{0:X8}" -f ([uint32]($LASTEXITCODE + 4294967296))
            } else {
                $hexCode = "0x{0:X8}" -f $LASTEXITCODE
            }
            $errorType = switch ($LASTEXITCODE) {
                -1073740791 { "Application crash (stack buffer overrun)" }
                -1073741819 { "Access violation" }
                -1073741571 { "Stack overflow" }
                -1073741515 { "DLL not found" }
                default { "Unknown error" }
            }
            
            Write-ColorOutput "   Transcription failed with exit code: $LASTEXITCODE ($hexCode) - $errorType" 'Error'
            
            # Show captured output/error
            if ($result -and $result.Count -gt 0) {
                Write-ColorOutput "   Output/Error details:" 'Error'
                foreach ($line in $result) {
                    if ($line.ToString().Trim()) {
                        Write-ColorOutput "     $line" 'Error'
                    }
                }
            } else {
                Write-ColorOutput "   No output captured - process crashed before producing output" 'Error'
            }
            
            $failed++
        }
    }
    catch {
        Write-ColorOutput "   Exception occurred: $($_.Exception.Message)" 'Error'
        $failed++
    }
    
    Write-Host ""
}

# Summary
$endTime = Get-Date
$totalTime = $endTime - $startTime

Write-ColorOutput "Batch transcription completed!" 'Success'
Write-ColorOutput "Summary:" 'Info'
Write-ColorOutput "   Successful: $successful" 'Success'
Write-ColorOutput "   Failed: $failed" $(if ($failed -gt 0) { 'Error' } else { 'Info' })
Write-ColorOutput "   Skipped: $skipped" $(if ($skipped -gt 0) { 'Warning' } else { 'Info' })
Write-ColorOutput "   Output directory: $(if ($OutputDir -eq '.') { 'current directory' } else { "$OutputDir\" })" 'Info'
Write-ColorOutput "   Total time: $(Format-Duration $totalTime)" 'Info'

if ($successful -gt 0) {
    Write-ColorOutput "$successful transcript(s) saved to: $(if ($OutputDir -eq '.') { 'current directory' } else { "$OutputDir\" })" 'Success'
}

if ($failed -gt 0) {
    Write-ColorOutput "$failed file(s) failed to process. Check the error messages above." 'Warning'
    exit 1
} else {
    exit 0
}
