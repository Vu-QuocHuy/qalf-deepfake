<#
.SYNOPSIS
Kịch bản chạy thí nghiệm đối chứng YuNet trên Server (PyTorch) cho video thô trên môi trường Windows.
Thực hiện cả hai điều kiện: C1 (CPU) và C2 (GPU).
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ProjectRoot

# Thiết lập đường dẫn cơ bản
$WindowsProjectRoot = "E:\DeepFakeData"
$StorageRoot = if ($env:QALF_STORAGE_ROOT) { $env:QALF_STORAGE_ROOT } else { $WindowsProjectRoot }
$Python = ".venv\Scripts\python.exe"

if (-Not (Test-Path $Python)) {
    Write-Host "ERROR: Python environment not found at $Python" -ForegroundColor Red
    exit 1
}

# Đường dẫn đến dữ liệu Celeb-DF-v2
$Manifest = if ($env:QALF_TEST_MANIFEST) { $env:QALF_TEST_MANIFEST } else { "F:\DeepFakedata\Celeb_DFv2\List_of_testing_videos.txt" }
$VideoRoot = if ($env:QALF_TEST_VIDEO_ROOT) { $env:QALF_TEST_VIDEO_ROOT } else { "F:\DeepFakedata\Celeb_DFv2" }

# Tham số mô hình
$Seed = if ($env:QALF_SEED) { $env:QALF_SEED } else { "42" }
$Checkpoint = if ($env:QALF_TEST_CHECKPOINT) { $env:QALF_TEST_CHECKPOINT } else { "$StorageRoot\experiments\ablation\baseline_seed${Seed}\best.pt" }

# Thư mục đầu ra
$OutputDirGpu = if ($env:QALF_TEST_OUTPUT_DIR) { $env:QALF_TEST_OUTPUT_DIR } else { "$StorageRoot\experiments\qalf_server_yunet_seed${Seed}_gpu" }
$OutputDirCpu = if ($env:QALF_TEST_OUTPUT_DIR) { $env:QALF_TEST_OUTPUT_DIR } else { "$StorageRoot\experiments\qalf_server_yunet_seed${Seed}_cpu" }

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " CHẠY THÍ NGHIỆM ĐỐI CHỨNG: SERVER + YUNET + PYTORCH" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "Manifest: $Manifest"
Write-Host "Video Root: $VideoRoot"
Write-Host "Checkpoint: $Checkpoint"

if (-Not (Test-Path $Checkpoint)) {
    Write-Host "ERROR: Checkpoint not found at $Checkpoint" -ForegroundColor Red
    exit 1
}

if (-Not (Test-Path $Manifest)) {
    Write-Host "ERROR: Manifest not found at $Manifest" -ForegroundColor Red
    exit 1
}

# Chạy C2: PyTorch GPU
Write-Host "`n>>> Bắt đầu chạy C2: PyTorch GPU" -ForegroundColor Yellow
if (Test-Path "$OutputDirGpu\metrics.json") {
    Write-Host "Đã có kết quả GPU tại $OutputDirGpu, bỏ qua..." -ForegroundColor Green
} else {
    & $Python scripts/evaluate_server_yunet_e2e.py `
        --manifest "$Manifest" `
        --video-root "$VideoRoot" `
        --checkpoint "$Checkpoint" `
        --output-dir "$OutputDirGpu" `
        --device "cuda" `
        --clips-per-video 3 `
        --texture-frames 8
}

# Chạy C1: PyTorch CPU
Write-Host "`n>>> Bắt đầu chạy C1: PyTorch CPU" -ForegroundColor Yellow
if (Test-Path "$OutputDirCpu\metrics.json") {
    Write-Host "Đã có kết quả CPU tại $OutputDirCpu, bỏ qua..." -ForegroundColor Green
} else {
    & $Python scripts/evaluate_server_yunet_e2e.py `
        --manifest "$Manifest" `
        --video-root "$VideoRoot" `
        --checkpoint "$Checkpoint" `
        --output-dir "$OutputDirCpu" `
        --device "cpu" `
        --clips-per-video 3 `
        --texture-frames 8
}

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "HOÀN THÀNH!" -ForegroundColor Green
Write-Host "Vui lòng kiểm tra metrics.json trong các thư mục đầu ra để lấy AUC."
