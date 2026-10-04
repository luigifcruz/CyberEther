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
$SigningMetadata = EnvOrDefault 'AZURE_SIGNING_METADATA' ''
$RequireSigning = (EnvOrDefault 'REQUIRE_SIGNING' '0') -eq '1'
$SigningArgs = @()

if ($RequireSigning -and [string]::IsNullOrWhiteSpace($SigningMetadata)) {
    Die 'REQUIRE_SIGNING=1 requires AZURE_SIGNING_METADATA'
}
if (![string]::IsNullOrWhiteSpace($SigningMetadata)) {
    $SigningMetadata = AbsolutePath $SigningMetadata
    if (!(Test-Path -LiteralPath $SigningMetadata -PathType Leaf)) {
        Die "signing metadata does not exist: $SigningMetadata"
    }
    $Metadata = Get-Content -Raw -LiteralPath $SigningMetadata | ConvertFrom-Json
    foreach ($Field in @('Endpoint', 'CodeSigningAccountName', 'CertificateProfileName')) {
        if ([string]::IsNullOrWhiteSpace($Metadata.$Field)) {
            Die "signing metadata is missing $Field"
        }
    }
    $SigningArgs = @('--azureTrustedSignFile', $SigningMetadata)
}

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
    --instLocation PerUser `
    @SigningArgs
if ($LASTEXITCODE -ne 0) {
    Die 'packaging CLI failed to create the Windows release'
}

if ($SigningArgs.Count -gt 0) {
    function AssertSigned($Path) {
        $Signature = Get-AuthenticodeSignature -LiteralPath $Path
        if ($Signature.Status -ne 'Valid') {
            Die "invalid signature on ${Path}: $($Signature.Status)"
        }
        if ($null -eq $Signature.TimeStamperCertificate) {
            Die "missing signature timestamp on $Path"
        }
    }

    $Installers = @(Get-ChildItem -LiteralPath $OutputDir -Filter '*Setup.exe' -File)
    $Packages = @(Get-ChildItem -LiteralPath $OutputDir -Filter '*-full.nupkg' -File)
    if ($Installers.Count -eq 0 -or $Packages.Count -eq 0) {
        Die 'expected a signed setup executable and a full update package'
    }
    foreach ($Installer in $Installers) {
        AssertSigned $Installer.FullName
    }

    # Check the shipped package, not the input directory: Velopack signs temporary copies.
    foreach ($Package in $Packages) {
        $VerifyDir = Join-Path ([System.IO.Path]::GetTempPath()) ([guid]::NewGuid().ToString())
        try {
            [System.IO.Compression.ZipFile]::ExtractToDirectory($Package.FullName, $VerifyDir)
            $Binaries = @(Get-ChildItem -LiteralPath $VerifyDir -Recurse -File |
                Where-Object { $_.Extension -in @('.exe', '.dll') })
            foreach ($Required in @($ExecutableName, 'jetstream.dll')) {
                if ($Required -notin $Binaries.Name) {
                    Die "missing $Required in $($Package.Name)"
                }
            }
            foreach ($Binary in $Binaries) {
                AssertSigned $Binary.FullName
            }
        } finally {
            if (Test-Path -LiteralPath $VerifyDir) {
                Remove-Item -LiteralPath $VerifyDir -Recurse -Force
            }
        }
    }
    Write-Host 'Verified signed and timestamped Windows release binaries.'
}

Write-Host "Created Windows release in: $OutputDir"
