param(
    [string]$PythonPath = "$PSScriptRoot/../.venv/Scripts/python.exe",
    [string]$LiveScript = "$PSScriptRoot/../live/demo_mt5.py",
    [string]$WorkingDirectory = "$PSScriptRoot/..",
    [switch]$NoKill,
    [switch]$VerboseLogging
)

function Write-Status {
    param(
        [string]$Message,
        [ConsoleColor]$Color = [ConsoleColor]::Gray
    )
    $timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $originalColor = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Host "[$timestamp] $Message"
    $Host.UI.RawUI.ForegroundColor = $originalColor
}

function Resolve-AbsolutePath {
    param([string]$PathValue)
    return (Resolve-Path -Path $PathValue).Path
}

try {
    $pythonPath = Resolve-AbsolutePath $PythonPath
    $liveScriptPath = Resolve-AbsolutePath $LiveScript
    $workingDir = Resolve-AbsolutePath $WorkingDirectory
} catch {
    Write-Status "Failed to resolve paths: $_" ([ConsoleColor]::Red)
    exit 1
}

Write-Status "Python executable: $pythonPath" ([ConsoleColor]::Cyan)
Write-Status "Live script: $liveScriptPath" ([ConsoleColor]::Cyan)
Write-Status "Working directory: $workingDir" ([ConsoleColor]::Cyan)

if (-not $NoKill) {
    try {
        $processes = Get-CimInstance Win32_Process | Where-Object {
            $_.Name -match 'python' -and $_.CommandLine -like "*${liveScriptPath}*"
        }
        foreach ($proc in $processes) {
            Write-Status "Stopping existing run (PID=$($proc.ProcessId))" ([ConsoleColor]::Yellow)
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        }
    } catch {
        Write-Status "Failed to stop existing processes: $_" ([ConsoleColor]::Red)
    }
} else {
    Write-Status "NoKill flag set; skipping process termination" ([ConsoleColor]::Yellow)
}

$arguments = "`"$liveScriptPath`""
if ($VerboseLogging) {
    $arguments += " --verbose"
}

try {
    Write-Status "Launching live bot..." ([ConsoleColor]::Green)
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $pythonPath
    $startInfo.Arguments = $arguments
    $startInfo.WorkingDirectory = $workingDir
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $false
    $startInfo.RedirectStandardError = $false

    $process = [System.Diagnostics.Process]::Start($startInfo)
    if ($null -eq $process) {
        Write-Status "Failed to start live bot." ([ConsoleColor]::Red)
        exit 1
    }
    Write-Status "Live bot started (PID=$($process.Id))" ([ConsoleColor]::Green)
} catch {
    Write-Status "Failed to start live bot: $_" ([ConsoleColor]::Red)
    exit 1
}

exit 0
