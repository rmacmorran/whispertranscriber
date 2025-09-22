# Example Batch Transcription Usage
# Simple examples of how to use batch-transcribe.ps1

Write-Host "🎤 Whisper Transcriber - Batch Processing Examples" -ForegroundColor Green
Write-Host ""

Write-Host "💡 Basic Usage Examples:" -ForegroundColor Cyan
Write-Host ""

Write-Host "1. Process all files in current folder (basic):" -ForegroundColor Yellow
Write-Host "   .\batch-transcribe.ps1" -ForegroundColor White
Write-Host ""

Write-Host "2. Process all files with timestamps:" -ForegroundColor Yellow  
Write-Host "   .\batch-transcribe.ps1 -Timestamps" -ForegroundColor White
Write-Host ""

Write-Host "3. Use different model (better quality, slower):" -ForegroundColor Yellow
Write-Host "   .\batch-transcribe.ps1 -Model small" -ForegroundColor White
Write-Host ""

Write-Host "4. Custom output directory:" -ForegroundColor Yellow
Write-Host "   .\batch-transcribe.ps1 -OutputDir 'my_transcripts'" -ForegroundColor White
Write-Host ""

Write-Host "5. Force overwrite existing transcripts:" -ForegroundColor Yellow
Write-Host "   .\batch-transcribe.ps1 -Force" -ForegroundColor White
Write-Host ""

Write-Host "6. Advanced: Small model, timestamps, custom output:" -ForegroundColor Yellow
Write-Host "   .\batch-transcribe.ps1 -Model small -Timestamps -OutputDir 'final_transcripts'" -ForegroundColor White
Write-Host ""

Write-Host "7. Get help:" -ForegroundColor Yellow
Write-Host "   .\batch-transcribe.ps1 -Help" -ForegroundColor White
Write-Host ""

Write-Host "📁 Supported File Types:" -ForegroundColor Cyan
Write-Host "   Video: .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm, .m4v" -ForegroundColor White
Write-Host "   Audio: .mp3, .wav, .flac, .aac, .ogg, .m4a, .wma" -ForegroundColor White
Write-Host ""

Write-Host "🚀 Quick Start:" -ForegroundColor Green
Write-Host "   1. Copy video/audio files to current folder" -ForegroundColor White
Write-Host "   2. Run: .\batch-transcribe.ps1" -ForegroundColor White
Write-Host "   3. Transcripts will be saved in 'transcripts' folder" -ForegroundColor White
