<#
.SYNOPSIS
  Install weekday Windows Task Scheduler jobs for Options Desk (local PC automation).
.DESCRIPTION
  Registers two non-intrusive weekday tasks under \OptionsDesk\:
    1. MorningDispatch (08:45 Mon-Fri): runs `opt notify --morning`
    2. PostCloseRefresh (18:15 Mon-Fri): runs `opt refresh` + archives 30 liquid chains + `opt notify --check`
  Run from PowerShell:
    powershell -ExecutionPolicy Bypass -File scripts\install_windows_tasks.ps1
  Remove anytime:
    powershell -ExecutionPolicy Bypass -File scripts\install_windows_tasks.ps1 -Uninstall
#>
param(
    [switch]$Uninstall
)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PyExe = Join-Path $RepoRoot "venv\Scripts\python.exe"
if (-not (Test-Path $PyExe)) {
    $PyExe = "python"
}

$MorningName = "OptionsDesk_MorningDispatch"
$RefreshName = "OptionsDesk_PostCloseRefresh"

if ($Uninstall) {
    Unregister-ScheduledTask -TaskName $MorningName -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $RefreshName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed OptionsDesk Windows Scheduled Tasks."
    exit 0
}

$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

$MorningAction = New-ScheduledTaskAction -Execute $PyExe -Argument "-m src.cli.main notify --morning" -WorkingDirectory $RepoRoot
$MorningTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 8:45AM
Register-ScheduledTask -TaskName $MorningName -Action $MorningAction -Trigger $MorningTrigger -Settings $Settings -Force | Out-Null

$RefreshCmd = "-c `"import subprocess, sys; subprocess.run([sys.executable,'-m','src.cli.main','refresh']); subprocess.run([sys.executable,'-m','src.cli.main','alpha','archive-chains','--limit','30']); subprocess.run([sys.executable,'-m','src.cli.main','notify','--check'])`""
$RefreshAction = New-ScheduledTaskAction -Execute $PyExe -Argument $RefreshCmd -WorkingDirectory $RepoRoot
$RefreshTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 6:15PM
Register-ScheduledTask -TaskName $RefreshName -Action $RefreshAction -Trigger $RefreshTrigger -Settings $Settings -Force | Out-Null

Write-Host "Installed Windows Scheduled Tasks:"
Write-Host "  1. $MorningName  (Mon-Fri 08:45 AM -> Discord Morning Dispatch)"
Write-Host "  2. $RefreshName  (Mon-Fri 06:15 PM -> Prices + Chain Archive + Signal Alerts)"
