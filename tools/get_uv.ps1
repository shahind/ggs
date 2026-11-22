# Prints the path of a uv executable: the one on PATH, else a pinned release downloaded into ~\.ggs\tools
# (checked against its published SHA-256). Messages go to stderr; stdout is only the path.
param([string]$Dest = "")
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Version = "0.13.9"
$Asset = "uv-x86_64-pc-windows-msvc.zip"
$Sha = "edd61f7f5728a56cc62455386661860bdfc7bab65872f39b0c25bdcc1d8e1beb"

if (-not $Dest) {
    $onPath = Get-Command uv -ErrorAction SilentlyContinue
    if ($onPath) { Write-Output $onPath.Source; exit 0 }
    $Dest = Join-Path $HOME ".ggs\tools\uv-$Version"
}
$exe = Join-Path $Dest "uv.exe"
if (Test-Path $exe) { Write-Output $exe; exit 0 }

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$zip = Join-Path $Dest $Asset
[Console]::Error.WriteLine("Downloading uv $Version (Python installer, about 20 MB)...")
Invoke-WebRequest -UseBasicParsing -Uri "https://github.com/astral-sh/uv/releases/download/$Version/$Asset" -OutFile $zip
$got = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
if ($got -ne $Sha) { Remove-Item $zip; throw "uv download is damaged (sha256 $got)" }
Expand-Archive -Path $zip -DestinationPath $Dest -Force
Remove-Item $zip
Write-Output $exe
