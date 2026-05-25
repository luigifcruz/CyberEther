# CyberEther for Windows

This directory is packaging only. It wraps an already-built Windows `cyberether.exe` executable and `jetstream.dll` runtime into an MSI installer.

## Usage

Build the executable and shared library, then package them:

```powershell
meson setup build --vsenv -Ddefault_library=shared -Db_vscrt=static_from_buildtype -Dtests=false -Dexamples=false -Dremote=enabled
meson compile -C build cyberether
apps/windows/create-msi.ps1
```

The default output is:

```text
.dist/windows/CyberEther-<version>-<arch>.msi
```

The MSI installs `cyberether.exe` and `jetstream.dll` into `Program Files\CyberEther`, creates a Start Menu shortcut, and adds the install directory to the system `PATH` by default.

Use environment variables to override packaging inputs when needed:

```powershell
$env:CYBERETHER_BINARY = "build\cyberether.exe"
$env:JETSTREAM_DLL = "build\jetstream.dll"
$env:OUTPUT_DIR = ".dist\windows"
apps/windows/create-msi.ps1
```

The MSI uses `ADD_TO_PATH=1` by default. Set `ADD_TO_PATH=0` while packaging to build an MSI that does not modify the system `PATH`:

```powershell
$env:ADD_TO_PATH = "0"
apps/windows/create-msi.ps1
```

The same property can be overridden at install time:

```powershell
msiexec /i CyberEther-<version>-<arch>.msi ADD_TO_PATH=0
```

WiX Toolset is required. If `wix.exe` is not available on `PATH`, the script installs the WiX CLI with `dotnet tool install` under `.dist\windows\.tools\wix`.
