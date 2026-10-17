# Take one snapshot of all sources. Safe to rerun: pages are cached per snapshot date.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& "$root\.venv\Scripts\jobnlp.exe" harvest *>> "$root\logs\harvest_scheduled.log"
& "$root\.venv\Scripts\jobnlp.exe" crawl *>> "$root\logs\harvest_scheduled.log"
