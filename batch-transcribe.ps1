# Batch Transcribe Script
# Processes all video and audio files in the current directory with Whisper Transcriber

param(
    [string]$OutputDir = "transcripts",
    [switch]$Timestamps,
    [string]$Model = "base",
    [switch]$Force,
    [switch]$ContinueOnError,
    [switch]$Help
)

# Show help
if ($Help) {
    Write-Host @"
Batch Transcribe Script - Process all video/audio files with Whisper Transcriber

USAGE:
    .\batch-transcribe.ps1 [OPTIONS]

OPTIONS:
    -OutputDir <path>   Output directory for transcripts (default: 'transcripts')
    -Timestamps         Include timestamps in transcripts
    -Model <size>       Whisper model size (tiny/base/small/medium/large, default: 'base')
    -Force              Overwrite existing transcript files
    -ContinueOnError    Continue processing other files if one fails
    -Help               Show this help message

EXAMPLES:
    .\batch-transcribe.ps1                                    # Basic transcription
    .\batch-transcribe.ps1 -Timestamps                       # With timestamps
    .\batch-transcribe.ps1 -OutputDir "my_transcripts"       # Custom output folder
    .\batch-transcribe.ps1 -Model small -Timestamps -Force   # Advanced options

SUPPORTED FORMATS:
    Video: .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm, .m4v
    Audio: .mp3, .wav, .flac, .aac, .ogg, .m4a, .wma
"@
    exit 0
}

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

# Try environment variable first (if set)
if ($env:WHISPER_TRANSCRIBER_PATH -and (Test-Path $env:WHISPER_TRANSCRIBER_PATH)) {
    $scriptPath = $env:WHISPER_TRANSCRIBER_PATH
}
# Try current directory
elseif (Test-Path ".\whisper-transcriber.py") {
    $scriptPath = Resolve-Path ".\whisper-transcriber.py"
}
# Try the directory where this batch script is located
elseif (Test-Path (Join-Path $PSScriptRoot "whisper-transcriber.py")) {
    $scriptPath = Join-Path $PSScriptRoot "whisper-transcriber.py"
}
# Try common installation paths
elseif (Test-Path "C:\Users\rmacmorran\projects\whisper-transcriber\whisper-transcriber.py") {
    $scriptPath = "C:\Users\rmacmorran\projects\whisper-transcriber\whisper-transcriber.py"
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

# Build base command arguments
$baseArgs = @('-q')  # Quiet mode
if ($Timestamps) { $baseArgs += '-t' }
if ($Model -ne 'base') { $baseArgs += @('-m', $Model) }

# Process each file
$successful = 0
$failed = 0
$skipped = 0
$startTime = Get-Date

Write-ColorOutput "Starting batch transcription..." 'Progress'
Write-ColorOutput "Model: $Model" 'Info'
Write-ColorOutput "Timestamps: $(if ($Timestamps) { 'Yes' } else { 'No' })" 'Info'
Write-ColorOutput "Output: $OutputDir\" 'Info'
Write-Host ""

for ($i = 0; $i -lt $allFiles.Count; $i++) {
    $file = $allFiles[$i]
    $progress = [math]::Round(($i / $allFiles.Count) * 100, 1)
    
    # Generate output filename
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
    $outputFile = Join-Path $OutputDir "$baseName.txt"
    
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
        
        # Run transcription
        Write-ColorOutput "   Transcribing..." 'Info'
        $result = & python $scriptPath @fileArgs 2>&1
        
        $fileEndTime = Get-Date
        $processingTime = $fileEndTime - $fileStartTime
        
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
            Write-ColorOutput "   Transcription failed with exit code: $LASTEXITCODE" 'Error'
            if ($result) {
                Write-ColorOutput "   Error details: $result" 'Error'
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
Write-ColorOutput "   Output directory: $OutputDir\" 'Info'
Write-ColorOutput "   Total time: $(Format-Duration $totalTime)" 'Info'

if ($successful -gt 0) {
    Write-ColorOutput "$successful transcript(s) saved to: $OutputDir\" 'Success'
}

if ($failed -gt 0) {
    Write-ColorOutput "$failed file(s) failed to process. Check the error messages above." 'Warning'
    exit 1
} else {
    exit 0
}
