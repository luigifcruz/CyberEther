param(
  [Parameter(Mandatory = $true)]
  [string]$Binary,

  [string]$OutputDir = ".dist/artifacts",
  [string]$Version = "",
  [string]$Arch = "x86_64"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$BinaryPath = Resolve-Path $Binary
$OutputPath = (New-Item -ItemType Directory -Force -Path $OutputDir).FullName

function Get-ProjectVersion {
  $mesonBuild = Join-Path $RepoRoot "meson.build"
  $line = Select-String -Path $mesonBuild -Pattern "version: '([^']+)'" | Select-Object -First 1
  if ($null -eq $line) {
    return "0.0.0"
  }
  return $line.Matches[0].Groups[1].Value
}

if ([string]::IsNullOrWhiteSpace($Version)) {
  $Version = Get-ProjectVersion
}

$Version = $Version.TrimStart("v")
if ($Version -notmatch '^\d+\.\d+\.\d+(\.\d+)?$') {
  throw "Windows MSI versions must be numeric. Got: $Version"
}

switch ($Arch) {
  "amd64" { $Arch = "x86_64" }
  "x64" { $Arch = "x86_64" }
}

$WixArch = if ($Arch -eq "x86_64") { "x64" } else { $Arch }
$StageDir = Join-Path $RepoRoot ".dist/windows/stage"

if (Test-Path $StageDir) {
  Remove-Item -Recurse -Force $StageDir
}

New-Item -ItemType Directory -Force -Path $StageDir | Out-Null
Copy-Item -Path $BinaryPath -Destination (Join-Path $StageDir "cyberether.exe")
Copy-Item -Path (Join-Path $RepoRoot "LICENSE") -Destination (Join-Path $StageDir "LICENSE.txt")

$ZipPath = Join-Path $OutputPath "cyberether-windows-$Arch.zip"
if (Test-Path $ZipPath) {
  Remove-Item -Force $ZipPath
}
Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $ZipPath -Force

$Wix = Get-Command wix -ErrorAction SilentlyContinue
if ($null -eq $Wix) {
  Write-Warning "WiX CLI was not found; skipped MSI creation."
  exit 0
}

$MsiPath = Join-Path $OutputPath "cyberether-windows-$Arch.msi"
$WixArgs = @(
  "build",
  (Join-Path $RepoRoot "apps/windows/CyberEther.wxs"),
  "-dSourceDir=$StageDir",
  "-dVersion=$Version",
  "-arch",
  $WixArch,
  "-out",
  $MsiPath
)
& $Wix.Source @WixArgs
