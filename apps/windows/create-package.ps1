Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$DefaultPackagerVersion = '1.2.0'
$AppName = 'CyberEther'
$PackId = 'CyberEther'
$ExecutableName = 'cyberether.exe'
$Channel = 'win-x64'

$ScriptDir = Split-Path -Parent $PSCommandPath
$RootDir = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path

function Die($Message) {
    throw "error: $Message"
}

function EnvOrDefault($Name, $Default) {
    $Value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $Default
    }
    return $Value
}

function AbsolutePath($Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Path))
}

function ProjectVersion {
    $Content = Get-Content -Raw -Path (Join-Path $RootDir 'meson.build')
    if ($Content -match "version:\s*'([^']+)'") {
        return $Matches[1]
    }
    return ''
}

function ResolvePackager($OutputDir) {
    $Configured = EnvOrDefault 'PACKAGER' ''
    if (![string]::IsNullOrWhiteSpace($Configured)) {
        $Path = AbsolutePath $Configured
        if (!(Test-Path -LiteralPath $Path)) {
            Die "packaging CLI does not exist: $Path"
        }
        return $Path
    }

    $ToolDir = Join-Path $OutputDir '.tools\vpk'
    $VpkExe = Join-Path $ToolDir 'vpk.exe'
    $Version = EnvOrDefault 'PACKAGE_TOOL_VERSION' $DefaultPackagerVersion
    $VersionDir = Join-Path $ToolDir ".store\vpk\$Version"
    if ((Test-Path -LiteralPath $VpkExe) -and (Test-Path -LiteralPath $VersionDir)) {
        return $VpkExe
    }

    $Dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
    if ($null -eq $Dotnet) {
        Die '.NET 8 SDK is required to install the packaging CLI'
    }

    if (Test-Path -LiteralPath $ToolDir) {
        Remove-Item -Recurse -Force $ToolDir
    }
    New-Item -ItemType Directory -Force -Path $ToolDir | Out-Null
    & $Dotnet.Source tool install --tool-path $ToolDir vpk --version $Version | Out-Host
    if ($LASTEXITCODE -ne 0) {
        Die 'failed to install the packaging CLI'
    }
    if (!(Test-Path -LiteralPath $VpkExe) -or !(Test-Path -LiteralPath $VersionDir)) {
        Die "packaging CLI $Version was not installed correctly"
    }

    return $VpkExe
}

if ($args.Count -ne 0) {
    Die 'create-package.ps1 takes no arguments; configure it with environment variables'
}

$Version = EnvOrDefault 'VERSION' (ProjectVersion)
$CyberEtherBinary = AbsolutePath (EnvOrDefault 'CYBERETHER_BINARY' (Join-Path $RootDir 'build\cyberether.exe'))
$JetstreamDll = AbsolutePath (EnvOrDefault 'JETSTREAM_DLL' (Join-Path $RootDir 'build\jetstream.dll'))
$IconSource = AbsolutePath (EnvOrDefault 'ICON_SOURCE' (Join-Path $RootDir 'apps\windows\cyberether.ico'))
$OutputDir = AbsolutePath (EnvOrDefault 'OUTPUT_DIR' (Join-Path $RootDir '.dist\windows'))
$ReleaseNotes = EnvOrDefault 'RELEASE_NOTES' ''
$Aumid = EnvOrDefault 'AUMID' 'ltd.luigi.CyberEther'
$PackDir = Join-Path $OutputDir '.pack'

if ([string]::IsNullOrWhiteSpace($Version)) {
    Die 'cannot determine project version'
}
foreach ($Path in @($CyberEtherBinary, $JetstreamDll, $IconSource)) {
    if (!(Test-Path -LiteralPath $Path)) {
        Die "packaging input does not exist: $Path"
    }
}
if ([string]::IsNullOrWhiteSpace($ReleaseNotes)) {
    $ReleaseNotes = Join-Path $OutputDir '.release-notes.md'
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    [System.IO.File]::WriteAllText($ReleaseNotes, '')
} else {
    $ReleaseNotes = AbsolutePath $ReleaseNotes
    if (!(Test-Path -LiteralPath $ReleaseNotes -PathType Leaf)) {
        Die "release notes do not exist: $ReleaseNotes"
    }
}

if (Test-Path -LiteralPath $PackDir) {
    Remove-Item -Recurse -Force $PackDir
}
New-Item -ItemType Directory -Force -Path $PackDir | Out-Null
Copy-Item -LiteralPath $CyberEtherBinary -Destination (Join-Path $PackDir $ExecutableName)
Copy-Item -LiteralPath $JetstreamDll -Destination (Join-Path $PackDir 'jetstream.dll')

$Packager = ResolvePackager $OutputDir
& $Packager pack `
    --packId $PackId `
    --packVersion $Version `
    --packDir $PackDir `
    --mainExe $ExecutableName `
    --packTitle $AppName `
    --packAuthors 'Luigi Cruz' `
    --icon $IconSource `
    --outputDir $OutputDir `
    --channel $Channel `
    --runtime win-x64 `
    --releaseNotes $ReleaseNotes `
    --aumid $Aumid `
    --shortcuts 'Desktop,StartMenuRoot' `
    --noPortable `
    --msi `
    --instLocation PerUser
if ($LASTEXITCODE -ne 0) {
    Die 'packaging CLI failed to create the Windows release'
}

Write-Host "Created Windows release in: $OutputDir"
