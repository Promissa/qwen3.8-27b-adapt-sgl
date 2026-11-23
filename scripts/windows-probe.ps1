# Read-only host diagnostics for the WSL memory and GPU residency gates.
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$result = [ordered]@{ recorded_at = (Get-Date).ToUniversalTime().ToString("o") }
try {
    $computer = Get-CimInstance Win32_ComputerSystem
    $result.physical_memory_bytes = $computer.TotalPhysicalMemory
    $result.video_controllers = @(Get-CimInstance Win32_VideoController | Select-Object Name, Status, CurrentHorizontalResolution, CurrentVerticalResolution)
} catch { $result.hardware_error = $_.Exception.Message }
try {
    $rows = @()
    foreach ($item in Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory) {
        $processName = "unknown"
        if ($item.Name -match "pid_(\d+)_") {
            try { $processName = (Get-Process -Id ([int]$Matches[1])).ProcessName } catch {}
        }
        $rows += [ordered]@{ counter_instance = $item.Name; process_name = $processName; dedicated_bytes = $item.DedicatedUsage; shared_bytes = $item.SharedUsage; committed_bytes = $item.TotalCommitted }
    }
    $result.gpu_process_memory = $rows
} catch { $result.gpu_counter_error = $_.Exception.Message }
$result | ConvertTo-Json -Depth 8 -Compress
