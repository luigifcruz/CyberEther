Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$AppName = 'CyberEther'
$Manufacturer = 'Luigi Cruz'
$ExecutableName = 'cyberether.exe'
$DefaultWixVersion = '6.0.2'
$WixUiExtension = 'WixToolset.UI.wixext'

$ScriptDir = Split-Path -Parent $PSCommandPath
$RootDir = (Resolve-Path (Join-Path $ScriptDir '..\..')).Path

function Die($Message) {
    throw "error: $Message"
}

function Msg($Message) {
    Write-Host $Message
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

function XmlEscape($Value) {
    return [System.Security.SecurityElement]::Escape([string]$Value)
}

function RtfEscape($Value) {
    $Builder = [System.Text.StringBuilder]::new()
    foreach ($Char in ([string]$Value).ToCharArray()) {
        $Code = [int][char]$Char
        if ($Char -eq '\') {
            [void]$Builder.Append('\\')
        } elseif ($Char -eq '{') {
            [void]$Builder.Append('\{')
        } elseif ($Char -eq '}') {
            [void]$Builder.Append('\}')
        } elseif ($Char -eq "`r") {
            continue
        } elseif ($Char -eq "`n") {
            [void]$Builder.Append('\par ')
        } elseif ($Code -lt 128) {
            [void]$Builder.Append($Char)
        } else {
            [void]$Builder.Append("\u$Code?")
        }
    }
    return $Builder.ToString()
}

function ProjectVersion {
    $MesonBuild = Join-Path $RootDir 'meson.build'
    $Content = Get-Content -Raw -Path $MesonBuild
    if ($Content -match "version:\s*'([^']+)'") {
        return $Matches[1]
    }
    return ''
}

function NormalizeArch($Value) {
    switch ($Value.ToLowerInvariant()) {
        'amd64' { return [pscustomobject]@{ Wix = 'x64'; Label = 'x86_64'; ProgramFiles = 'ProgramFiles64Folder' } }
        'x64' { return [pscustomobject]@{ Wix = 'x64'; Label = 'x86_64'; ProgramFiles = 'ProgramFiles64Folder' } }
        'x86_64' { return [pscustomobject]@{ Wix = 'x64'; Label = 'x86_64'; ProgramFiles = 'ProgramFiles64Folder' } }
        'aarch64' { return [pscustomobject]@{ Wix = 'arm64'; Label = 'arm64'; ProgramFiles = 'ProgramFiles64Folder' } }
        'arm64' { return [pscustomobject]@{ Wix = 'arm64'; Label = 'arm64'; ProgramFiles = 'ProgramFiles64Folder' } }
        default { Die "unsupported Windows MSI architecture: $Value" }
    }
}

function ResolveTool($Tool) {
    if ($Tool.Contains('\') -or $Tool.Contains('/')) {
        $Path = AbsolutePath $Tool
        if (!(Test-Path $Path)) {
            Die "tool does not exist: $Path"
        }
        return $Path
    }

    $Command = Get-Command $Tool -ErrorAction SilentlyContinue
    if ($null -eq $Command) {
        Die "$Tool is required"
    }
    return $Command.Source
}

function ResolveWixTool($Tool) {
    if ($Tool.Contains('\') -or $Tool.Contains('/')) {
        $Path = AbsolutePath $Tool
        if (!(Test-Path -LiteralPath $Path)) {
            Die "WiX path does not exist: $Path"
        }

        $Item = Get-Item -LiteralPath $Path
        if (!$Item.PSIsContainer) {
            return $Path
        }

        foreach ($Candidate in @((Join-Path $Path 'wix.exe'), (Join-Path $Path 'bin\wix.exe'))) {
            if (Test-Path -LiteralPath $Candidate) {
                return $Candidate
            }
        }

        Msg "Ignoring WiX directory without wix.exe: $Path"
        return ''
    }

    return ResolveTool $Tool
}

function ResolveWix($OutputDir) {
    $WixTool = EnvOrDefault 'WIX' ''
    if (![string]::IsNullOrWhiteSpace($WixTool)) {
        $ResolvedWixTool = ResolveWixTool $WixTool
        if (![string]::IsNullOrWhiteSpace($ResolvedWixTool)) {
            return $ResolvedWixTool
        }
    }

    $Existing = Get-Command wix -ErrorAction SilentlyContinue
    if ($null -ne $Existing) {
        return $Existing.Source
    }

    $ToolDir = Join-Path $OutputDir '.tools\wix'
    $WixExe = Join-Path $ToolDir 'wix.exe'
    if (!(Test-Path $WixExe)) {
        $Dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
        if ($null -eq $Dotnet) {
            Die 'dotnet is required to install the WiX Toolset CLI'
        }

        New-Item -ItemType Directory -Force -Path $ToolDir | Out-Null

        $InstallArgs = @('tool', 'install', '--tool-path', $ToolDir, 'wix')
        $WixVersion = EnvOrDefault 'WIX_VERSION' $DefaultWixVersion
        if (![string]::IsNullOrWhiteSpace($WixVersion)) {
            $InstallArgs += @('--version', $WixVersion)
        }

        Msg "Installing WiX Toolset CLI $WixVersion"
        $InstallOutput = & $Dotnet.Source @InstallArgs 2>&1
        $InstallExitCode = $LASTEXITCODE
        $InstallOutput | ForEach-Object { Msg $_ }
        if ($InstallExitCode -ne 0) {
            Die 'failed to install WiX Toolset CLI'
        }
        if (!(Test-Path -LiteralPath $WixExe)) {
            Die "WiX Toolset CLI was not installed to $WixExe"
        }
    }

    return $WixExe
}

function GetWixVersion($WixPath) {
    $VersionOutput = & $WixPath --version 2>&1
    $VersionExitCode = $LASTEXITCODE
    $VersionText = [string]::Join("`n", @($VersionOutput))
    if ($VersionExitCode -ne 0) {
        Die 'failed to determine WiX Toolset CLI version'
    }
    if ($VersionText -match '([0-9]+\.[0-9]+\.[0-9]+)') {
        return $Matches[1]
    }
    Die "cannot determine WiX Toolset CLI version from: $VersionText"
}

function EnsureWixExtension($WixPath, $ExtensionRef) {
    Msg "Ensuring WiX extension: $ExtensionRef"
    $ExtensionOutput = & $WixPath extension add $ExtensionRef 2>&1
    $ExtensionExitCode = $LASTEXITCODE
    $ExtensionOutput | ForEach-Object { Msg $_ }
    if ($ExtensionExitCode -ne 0) {
        $ExtensionText = [string]::Join("`n", @($ExtensionOutput))
        if ($ExtensionText -notmatch 'already') {
            Die "failed to add WiX extension: $ExtensionRef"
        }
    }
}

function ValidateMetadata($AppName, $Manufacturer, $ExecutableName, $Version, $UpgradeCode, $AddToPath, $CreateDesktopShortcut) {
    if ($AppName -notmatch '^[A-Za-z0-9][A-Za-z0-9._ -]*$') {
        Die 'APP_NAME contains unsupported characters'
    }
    if ($Manufacturer -notmatch '^[A-Za-z0-9][A-Za-z0-9._ -]*$') {
        Die 'MANUFACTURER contains unsupported characters'
    }
    if ($ExecutableName -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*\.exe$') {
        Die 'EXECUTABLE_NAME must be a Windows .exe filename'
    }
    if ($Version -notmatch '^[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?$') {
        Die 'VERSION must be an MSI-compatible numeric version'
    }
    if ($UpgradeCode -notmatch '^\{?[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}?$') {
        Die 'UPGRADE_CODE must be a GUID'
    }
    if ($AddToPath -ne '0' -and $AddToPath -ne '1') {
        Die 'ADD_TO_PATH must be 0 or 1'
    }
    if ($CreateDesktopShortcut -ne '0' -and $CreateDesktopShortcut -ne '1') {
        Die 'CREATE_DESKTOP_SHORTCUT must be 0 or 1'
    }
}

function ValidateInputs($CyberEtherBinary, $JetstreamDll, $IconSource) {
    if (!(Test-Path $CyberEtherBinary)) {
        Die "binary does not exist: $CyberEtherBinary"
    }
    if (!(Test-Path $JetstreamDll)) {
        Die "DLL does not exist: $JetstreamDll"
    }
    if (!(Test-Path $IconSource)) {
        Die "icon does not exist: $IconSource"
    }
}

function WriteLicenseRtf($Path) {
    $LicensePath = Join-Path $RootDir 'LICENSE'
    if (!(Test-Path $LicensePath)) {
        Die "license file does not exist: $LicensePath"
    }

    $LicenseText = Get-Content -Raw -Path $LicensePath
    $LicenseBody = RtfEscape $LicenseText
    $LicenseRtf = "{\rtf1\ansi\deff0{\fonttbl{\f0 Segoe UI;}}\f0\fs18 $LicenseBody}"
    Set-Content -Path $Path -Value $LicenseRtf -Encoding ASCII
}

function WriteWixSource($Path, $Data) {
    $AppNameEsc = XmlEscape $Data.AppName
    $ManufacturerEsc = XmlEscape $Data.Manufacturer
    $ExecutableNameEsc = XmlEscape $Data.ExecutableName
    $CyberEtherBinaryEsc = XmlEscape $Data.CyberEtherBinary
    $JetstreamDllEsc = XmlEscape $Data.JetstreamDll
    $IconSourceEsc = XmlEscape $Data.IconSource
    $LicenseRtfEsc = XmlEscape $Data.LicenseRtf
    $VersionEsc = XmlEscape $Data.Version
    $UpgradeCodeEsc = XmlEscape $Data.UpgradeCode.Trim('{}')
    $AddToPathEsc = XmlEscape $Data.AddToPath
    $CreateDesktopShortcutEsc = XmlEscape $Data.CreateDesktopShortcut
    $ProgramFilesFolderEsc = XmlEscape $Data.ProgramFilesFolder

    $Wxs = @"
<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs" xmlns:ui="http://wixtoolset.org/schemas/v4/wxs/ui">
  <Package Name="$AppNameEsc" Manufacturer="$ManufacturerEsc" Version="$VersionEsc" UpgradeCode="$UpgradeCodeEsc" Scope="perMachine">
    <MajorUpgrade DowngradeErrorMessage="A newer version of $AppNameEsc is already installed." />
    <MediaTemplate EmbedCab="yes" />
    <Property Id="ADD_TO_PATH" Value="$AddToPathEsc" />
    <Property Id="CREATE_DESKTOP_SHORTCUT" Value="$CreateDesktopShortcutEsc" />
    <Icon Id="CyberEtherIcon" SourceFile="$IconSourceEsc" />
    <Property Id="ARPPRODUCTICON" Value="CyberEtherIcon" />
    <ui:WixUI Id="WixUI_FeatureTree" />
    <WixVariable Id="WixUILicenseRtf" Value="$LicenseRtfEsc" />

    <StandardDirectory Id="$ProgramFilesFolderEsc">
      <Directory Id="INSTALLFOLDER" Name="$AppNameEsc">
        <Component Id="CyberEtherExecutable" Guid="*">
          <File Id="CyberEtherExecutableFile" Source="$CyberEtherBinaryEsc" KeyPath="yes" />
        </Component>
        <Component Id="JetstreamRuntime" Guid="*">
          <File Id="JetstreamRuntimeFile" Source="$JetstreamDllEsc" KeyPath="yes" />
        </Component>
        <Component Id="PathEnvironment" Guid="*">
          <RegistryValue Root="HKLM" Key="Software\$AppNameEsc" Name="PathEnvironment" Type="integer" Value="1" KeyPath="yes" />
          <Environment Id="CyberEtherPath" Name="PATH" Action="set" Part="last" System="yes" Permanent="no" Value="[INSTALLFOLDER]" />
        </Component>
      </Directory>
    </StandardDirectory>

    <StandardDirectory Id="ProgramMenuFolder">
      <Directory Id="ApplicationProgramsFolder" Name="$AppNameEsc">
        <Component Id="StartMenuShortcut" Guid="*">
          <Shortcut Id="ApplicationStartMenuShortcut" Name="$AppNameEsc" Description="CyberEther signal processing runtime" Target="[INSTALLFOLDER]$ExecutableNameEsc" WorkingDirectory="INSTALLFOLDER" Icon="CyberEtherIcon" />
          <RemoveFolder Id="ApplicationProgramsFolder" On="uninstall" />
          <RegistryValue Root="HKLM" Key="Software\$AppNameEsc" Name="StartMenuShortcut" Type="integer" Value="1" KeyPath="yes" />
        </Component>
      </Directory>
    </StandardDirectory>

    <StandardDirectory Id="DesktopFolder">
      <Component Id="DesktopShortcut" Guid="*">
        <Shortcut Id="ApplicationDesktopShortcut" Name="$AppNameEsc" Description="CyberEther signal processing runtime" Target="[INSTALLFOLDER]$ExecutableNameEsc" WorkingDirectory="INSTALLFOLDER" Icon="CyberEtherIcon" />
        <RegistryValue Root="HKLM" Key="Software\$AppNameEsc" Name="DesktopShortcut" Type="integer" Value="1" KeyPath="yes" />
      </Component>
    </StandardDirectory>

    <Feature Id="DefaultFeature" Title="$AppNameEsc" Description="Install CyberEther and required runtime files." Level="1" Display="expand">
      <ComponentRef Id="CyberEtherExecutable" />
      <ComponentRef Id="JetstreamRuntime" />
      <ComponentRef Id="StartMenuShortcut" />
      <Feature Id="DesktopShortcutFeature" Title="Desktop shortcut" Description="Create a desktop shortcut." Level="1">
        <Level Value="0" Condition='CREATE_DESKTOP_SHORTCUT = "0"' />
        <ComponentRef Id="DesktopShortcut" />
      </Feature>
      <Feature Id="PathEnvironmentFeature" Title="Add to PATH" Description="Add the installation folder to the system PATH." Level="1">
        <Level Value="0" Condition='ADD_TO_PATH = "0"' />
        <ComponentRef Id="PathEnvironment" />
      </Feature>
    </Feature>
  </Package>
</Wix>
"@

    Set-Content -Path $Path -Value $Wxs -Encoding UTF8
}

function CreateMsi($WxsPath, $MsiPath, $WixPath, $WixArch, $WixExtension) {
    if (Test-Path $MsiPath) {
        Remove-Item -Force $MsiPath
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $MsiPath) | Out-Null

    & $WixPath build $WxsPath -ext $WixExtension -arch $WixArch -o $MsiPath
    if ($LASTEXITCODE -ne 0) {
        Die 'WiX failed to create the MSI'
    }
    if (!(Test-Path $MsiPath)) {
        Die "MSI was not created: $MsiPath"
    }

    Msg "Created MSI: $MsiPath"
}

if ($args.Count -ne 0) {
    Die 'create-msi.ps1 takes no arguments; configure it with environment variables'
}

$AppName = EnvOrDefault 'APP_NAME' $AppName
$Manufacturer = EnvOrDefault 'MANUFACTURER' $Manufacturer
$ExecutableName = EnvOrDefault 'EXECUTABLE_NAME' $ExecutableName
$Version = EnvOrDefault 'VERSION' ''
$ArchValue = EnvOrDefault 'ARCH' ([System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString())
$CyberEtherBinary = EnvOrDefault 'CYBERETHER_BINARY' (Join-Path $RootDir 'build\cyberether.exe')
$JetstreamDll = EnvOrDefault 'JETSTREAM_DLL' (Join-Path $RootDir 'build\jetstream.dll')
$IconSource = EnvOrDefault 'ICON_SOURCE' (Join-Path $RootDir 'apps\windows\cyberether.ico')
$OutputDir = EnvOrDefault 'OUTPUT_DIR' (Join-Path $RootDir '.dist\windows')
$UpgradeCode = EnvOrDefault 'UPGRADE_CODE' '0D57D910-3A13-4E53-9B8F-0F4A98F7F7EC'
$AddToPath = EnvOrDefault 'ADD_TO_PATH' '1'
$CreateDesktopShortcut = EnvOrDefault 'CREATE_DESKTOP_SHORTCUT' '1'

if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = ProjectVersion
}
if ([string]::IsNullOrWhiteSpace($Version)) {
    Die 'cannot determine project version'
}

$Arch = NormalizeArch $ArchValue
ValidateMetadata $AppName $Manufacturer $ExecutableName $Version $UpgradeCode $AddToPath $CreateDesktopShortcut

$CyberEtherBinary = AbsolutePath $CyberEtherBinary
$JetstreamDll = AbsolutePath $JetstreamDll
$IconSource = AbsolutePath $IconSource
$OutputDir = AbsolutePath $OutputDir
$MsiPath = AbsolutePath (EnvOrDefault 'MSI_PATH' (Join-Path $OutputDir "$AppName-$Version-$($Arch.Label).msi"))
$WorkDir = Join-Path $OutputDir '.msi-work'
$WxsPath = Join-Path $WorkDir 'CyberEther.wxs'
$LicenseRtfPath = Join-Path $WorkDir 'LICENSE.rtf'

ValidateInputs $CyberEtherBinary $JetstreamDll $IconSource
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null

$WixPath = ResolveWix $OutputDir
$WixUiExtensionVersion = EnvOrDefault 'WIX_UI_EXTENSION_VERSION' (GetWixVersion $WixPath)
EnsureWixExtension $WixPath "$WixUiExtension/$WixUiExtensionVersion"
WriteLicenseRtf $LicenseRtfPath
$WixData = [pscustomobject]@{
    AppName = $AppName
    Manufacturer = $Manufacturer
    ExecutableName = $ExecutableName
    Version = $Version
    UpgradeCode = $UpgradeCode
    AddToPath = $AddToPath
    CreateDesktopShortcut = $CreateDesktopShortcut
    ProgramFilesFolder = $Arch.ProgramFiles
    CyberEtherBinary = $CyberEtherBinary
    JetstreamDll = $JetstreamDll
    IconSource = $IconSource
    LicenseRtf = $LicenseRtfPath
}

WriteWixSource $WxsPath $WixData
CreateMsi $WxsPath $MsiPath $WixPath $Arch.Wix $WixUiExtension
