# Registers a weekly snapshot task in Windows Task Scheduler (run once yourself, in PowerShell).
# The build never runs this file. To remove the task later:
#   Unregister-ScheduledTask -TaskName "jobnlp-weekly-harvest" -Confirm:$false
$root = Split-Path -Parent $PSScriptRoot
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$root\scripts\harvest.ps1`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 9am
# StartWhenAvailable: a run missed while the laptop slept starts at the next wake.
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 12)
Register-ScheduledTask -TaskName "jobnlp-weekly-harvest" -Action $action -Trigger $trigger `
    -Settings $settings -Description "Weekly job posting snapshot for job-market-nlp"
