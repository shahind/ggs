# Installs the Claude Code agents, skills and settings into .claude\
# Run once from the project root:  .\setup-claude.ps1
# (If PowerShell blocks scripts:  powershell -ExecutionPolicy Bypass -File .\setup-claude.ps1)
$ErrorActionPreference = "Stop"
$src = Join-Path $PSScriptRoot "claude-config"
$dst = Join-Path $PSScriptRoot ".claude"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Copy-Item -Path (Join-Path $src "*") -Destination $dst -Recurse -Force
Write-Host "Installed agents, skills and settings into $dst"
Get-ChildItem -Path (Join-Path $dst "agents") -Name
