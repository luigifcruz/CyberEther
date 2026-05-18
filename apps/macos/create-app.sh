#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
PATH="/usr/bin:/bin:/usr/sbin:/sbin"

APP_NAME="${APP_NAME:-CyberEther}"
BUNDLE_ID="${BUNDLE_ID:-ltd.luigi.CyberEther}"
EXECUTABLE_NAME="${EXECUTABLE_NAME:-cyberether}"
MIN_MACOS="${MIN_MACOS:-12.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION="${VERSION:-}"
CYBERETHER_BINARY="${CYBERETHER_BINARY:-$ROOT_DIR/build/cyberether}"
ICON_SOURCE="${ICON_SOURCE:-$ROOT_DIR/resources/assets/icon.png}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/.dist/macos}"

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
    [[ "$EXECUTABLE_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "EXECUTABLE_NAME contains unsupported characters"
    [[ "$BUNDLE_ID" =~ ^[A-Za-z0-9][A-Za-z0-9.-]*[A-Za-z0-9]$ ]] || die "BUNDLE_ID contains unsupported characters"
    [[ "$MIN_MACOS" =~ ^[0-9]+(\.[0-9]+){1,2}$ ]] || die "MIN_MACOS must be a macOS version"
}

escape_sed() {
    printf '%s' "$1" | sed -e 's/[\/&|]/\\&/g'
}

project_version() {
    awk -F"'" '/version:/ { print $2; exit }' "$ROOT_DIR/meson.build" 2>/dev/null || true
}

if [[ $# -ne 0 ]]; then
    die "create-app.sh takes no arguments; configure it with environment variables"
fi

if [[ -z "$VERSION" ]]; then
    VERSION="$(project_version)"
fi

if [[ -z "$VERSION" ]]; then
    die "cannot determine project version"
fi

[[ "$VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || die "VERSION contains unsupported characters"
validate_metadata

CYBERETHER_BINARY="$(abs_path "$CYBERETHER_BINARY")"
ICON_SOURCE="$(abs_path "$ICON_SOURCE")"
OUTPUT_DIR="$(abs_path "$OUTPUT_DIR")"

APP_PATH="$(abs_path "${APP_PATH:-$OUTPUT_DIR/$APP_NAME.app}")"
CONTENTS_DIR="$APP_PATH/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"
WORK_DIR="$OUTPUT_DIR/.app-work"

validate_inputs() {
    [[ -f "$CYBERETHER_BINARY" ]] || die "binary does not exist: $CYBERETHER_BINARY"
    [[ -f "$ICON_SOURCE" ]] || die "icon source does not exist: $ICON_SOURCE"
}

render_plist() {
    sed \
        -e "s|@APP_NAME@|$(escape_sed "$APP_NAME")|g" \
        -e "s|@EXECUTABLE_NAME@|$(escape_sed "$EXECUTABLE_NAME")|g" \
        -e "s|@BUNDLE_ID@|$(escape_sed "$BUNDLE_ID")|g" \
        -e "s|@VERSION@|$(escape_sed "$VERSION")|g" \
        -e "s|@MIN_MACOS@|$(escape_sed "$MIN_MACOS")|g" \
        "$SCRIPT_DIR/Info.plist.in" > "$CONTENTS_DIR/Info.plist"
}

generate_icon() {
    local iconset="$WORK_DIR/AppIcon.iconset"

    require_tool sips
    require_tool iconutil

    rm -rf "$iconset"
    mkdir -p "$iconset"

    for size in 16 32 128 256 512; do
        local double_size=$((size * 2))
        sips -z "$size" "$size" "$ICON_SOURCE" --out "$iconset/icon_${size}x${size}.png" >/dev/null
        sips -z "$double_size" "$double_size" "$ICON_SOURCE" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
    done

    iconutil -c icns "$iconset" -o "$RESOURCES_DIR/AppIcon.icns"
}

create_app() {
    rm -rf "$APP_PATH" "$WORK_DIR"
    mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$WORK_DIR"

    cp "$CYBERETHER_BINARY" "$MACOS_DIR/$EXECUTABLE_NAME"
    chmod 755 "$MACOS_DIR/$EXECUTABLE_NAME"
    printf 'APPL????' > "$CONTENTS_DIR/PkgInfo"

    render_plist
    generate_icon

    msg "Created app bundle: $APP_PATH"
}

validate_inputs
mkdir -p "$OUTPUT_DIR"
create_app
