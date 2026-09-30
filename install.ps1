# Sajag AI installer for Windows 11 on Snapdragon (Arm64).
# Usage: powershell -ExecutionPolicy Bypass -File install.ps1
$ErrorActionPreference = "Stop"
Write-Host "Sajag AI setup" -ForegroundColor Red
$arch = $env:PROCESSOR_ARCHITECTURE
if ($arch -ne "ARM64") { Write-Warning "This PC is $arch. The NPU path needs a Snapdragon (ARM64) PC; CPU fallback will be used." }
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
if ($arch -eq "ARM64") {
  Get-Content requirements-snapdragon.txt | Where-Object { $_ -notmatch "rapidocr" -and $_ -notmatch "^#" -and $_.Trim() } | ForEach-Object { pip install ($_ -split "#")[0].Trim() }
  pip install rapidocr-onnxruntime --no-deps
} else {
  pip install -r requirements.txt
}
python -m sajag fetch-models --chipset x_elite
python scripts\prepare_npu_models.py
python -m sajag doctor
Write-Host "Done. Start with:  .\.venv\Scripts\python -m sajag run" -ForegroundColor Green
