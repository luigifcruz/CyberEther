#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
PATH="/usr/bin:/bin:/usr/sbin:/sbin"

APP_NAME="${APP_NAME:-CyberEther}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION="${VERSION:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/.dist/macos}"
MOUNT_DIR=""

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

msg() {
    printf '%s\n' "$*"
}

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$PWD/${1#./}" ;;
    esac
}

require_tool() {
    command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

validate_metadata() {
    [[ "$APP_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._\ -]*$ ]] || die "APP_NAME contains unsupported characters"
}

project_version() {
    awk -F"'" '/version:/ { print $2; exit }' "$ROOT_DIR/meson.build" 2>/dev/null || true
}

cleanup() {
    if [[ -n "$MOUNT_DIR" && -d "$MOUNT_DIR" ]]; then
        hdiutil detach "$MOUNT_DIR" >/dev/null 2>&1 || true
        rmdir "$MOUNT_DIR" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

if [[ $# -ne 0 ]]; then
    die "create-dmg.sh takes no arguments; configure it with environment variables"
fi

if [[ -z "$VERSION" ]]; then
    VERSION="$(project_version)"
fi

if [[ -z "$VERSION" ]]; then
    die "cannot determine project version"
fi

[[ "$VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || die "VERSION contains unsupported characters"
validate_metadata

OUTPUT_DIR="$(abs_path "$OUTPUT_DIR")"
APP_PATH="$(abs_path "${APP_PATH:-$OUTPUT_DIR/$APP_NAME.app}")"
DMG_PATH="$(abs_path "${DMG_PATH:-$OUTPUT_DIR/$APP_NAME-$VERSION.dmg}")"
WORK_DIR="$OUTPUT_DIR/.dmg-work"

validate_inputs() {
    [[ -d "$APP_PATH" ]] || die "app bundle does not exist: $APP_PATH"
}

style_dmg() {
    local volume_path="$1"

    command -v osascript >/dev/null 2>&1 || return 0

    osascript <<APPLESCRIPT >/dev/null 2>&1 || msg "warning: Finder styling failed; DMG is still usable"
tell application "Finder"
    tell disk "$APP_NAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set bounds of container window to {120, 120, 780, 540}
        set theViewOptions to the icon view options of container window
        set arrangement of theViewOptions to not arranged
        set icon size of theViewOptions to 96
        set position of item "$APP_NAME.app" of container window to {165, 245}
        set position of item "Applications" of container window to {495, 245}
        update without registering applications
        delay 1
        close
    end tell
end tell
APPLESCRIPT

    command -v bless >/dev/null 2>&1 && bless --folder "$volume_path" --openfolder "$volume_path" >/dev/null 2>&1 || true
}

detach_dmg() {
    local path="$1"

    for _ in 1 2 3 4 5; do
        if hdiutil detach "$path" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done

    hdiutil detach -force "$path" >/dev/null
}

create_dmg() {
    local dmg_root="$WORK_DIR/dmg-root"
    local temp_dmg="$WORK_DIR/$APP_NAME-rw.dmg"
    local dmg_size_mb

    require_tool hdiutil
    require_tool ditto

    rm -rf "$dmg_root"
    mkdir -p "$dmg_root" "${DMG_PATH%/*}"
    ditto "$APP_PATH" "$dmg_root/$APP_NAME.app"
    ln -s /Applications "$dmg_root/Applications"

    dmg_size_mb="$(du -sm "$dmg_root" | awk '{ print $1 }')"
    dmg_size_mb=$((dmg_size_mb + 128))

    rm -f "$temp_dmg" "$DMG_PATH"
    hdiutil create -volname "$APP_NAME" -size "${dmg_size_mb}m" -fs HFS+ "$temp_dmg" >/dev/null

    MOUNT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cyberether-dmg.XXXXXX")"
    hdiutil attach "$temp_dmg" -mountpoint "$MOUNT_DIR" -nobrowse -noverify -noautoopen >/dev/null
    ditto "$dmg_root" "$MOUNT_DIR"
    style_dmg "$MOUNT_DIR"
    sync
    detach_dmg "$MOUNT_DIR"
    rmdir "$MOUNT_DIR" >/dev/null 2>&1 || true
    MOUNT_DIR=""

    hdiutil convert "$temp_dmg" -format UDZO -imagekey zlib-level=9 -o "$DMG_PATH" >/dev/null
    rm -f "$temp_dmg"

    msg "Created installer DMG: $DMG_PATH"
}

validate_inputs
mkdir -p "$OUTPUT_DIR"
create_dmg
