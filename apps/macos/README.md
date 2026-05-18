# CyberEther for macOS

This directory is packaging only. It does not build a separate native frontend; it wraps an already-built `cyberether` executable in a macOS `.app` bundle and creates a drag-to-install DMG.

## Usage

Build the executable, then package it. The packaging scripts take no arguments and are configured with environment variables:

```sh
meson setup build -Dtests=false -Dexamples=false
meson compile -C build cyberether
apps/macos/create-app.sh
apps/macos/create-dmg.sh
```

The default outputs are:

```text
.dist/macos/CyberEther.app
.dist/macos/CyberEther-<version>.dmg
```

## Packaging

The packaging flow is split into four stages:

```sh
apps/macos/create-app.sh
apps/macos/sign-notarize-app.sh
apps/macos/create-dmg.sh
apps/macos/sign-notarize-dmg.sh
```

The create stages only assemble the app bundle and DMG. The sign/notarize stages require Apple Developer ID and App Store Connect API credentials in environment variables. In GitHub Actions, push-triggered packaging signs and notarizes through the `macos-notarization` environment.

Required signing/notarization variables:

```text
APPLE_CERT_P12_BASE64
APPLE_CERT_PASSWORD
APPLE_CODESIGN_IDENTITY
APPLE_NOTARY_KEY_ID
APPLE_NOTARY_ISSUER_ID
APPLE_NOTARY_KEY_P8_BASE64
```

Use environment variables to override packaging inputs when needed:

```sh
export CYBERETHER_BINARY="build/cyberether"
export OUTPUT_DIR=".dist/macos"
apps/macos/create-app.sh
apps/macos/create-dmg.sh
```

The DMG contains `CyberEther.app` and an `/Applications` shortcut so users can install by drag and drop.
