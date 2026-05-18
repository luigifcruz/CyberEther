# CyberEther for macOS

This directory is packaging only. It does not build a separate native frontend; it wraps an already-built `cyberether` executable in a macOS `.app` bundle and creates a drag-to-install DMG.

## Usage

Build the executable, then package it. The packager takes no arguments and always creates a DMG:

```sh
meson setup build -Dtests=false -Dexamples=false
meson compile -C build cyberether
apps/macos/package.sh
```

The default outputs are:

```text
.dist/macos/CyberEther.app
.dist/macos/CyberEther-<version>.dmg
```

You can also build the DMG through Meson on macOS. This target is not built by a normal compile:

```sh
meson compile -C build cyberether-dmg
```

## Packaging

The script only assembles the app bundle and DMG. It does not sign or notarize artifacts.

Use environment variables to override packaging inputs when needed:

```sh
export CYBERETHER_BINARY="build/cyberether"
export OUTPUT_DIR=".dist/macos"
apps/macos/package.sh
```

The DMG contains `CyberEther.app` and an `/Applications` shortcut so users can install by drag and drop.
