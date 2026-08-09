#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

APP_NAME="${APP_NAME:-CyberEther}"
APP_ID="${APP_ID:-CyberEther}"
PACK_ID="${PACK_ID:-}"
EXECUTABLE_NAME="${EXECUTABLE_NAME:-cyberether}"
PACKAGE_TOOL_VERSION="${PACKAGE_TOOL_VERSION:-1.2.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION="${VERSION:-}"
ARCH="${ARCH:-}"
APPIMAGE_SUFFIX="${APPIMAGE_SUFFIX:-}"
CYBERETHER_BINARY="${CYBERETHER_BINARY:-$ROOT_DIR/build/cyberether}"
JETSTREAM_SO="${JETSTREAM_SO:-$ROOT_DIR/build/libjetstream.so}"
ICON_SOURCE="${ICON_SOURCE:-$ROOT_DIR/resources/assets/icon.png}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/.dist/linux}"
RELEASE_NOTES="${RELEASE_NOTES:-/dev/null}"
JETSTREAM_SO_NAME="libjetstream.so"

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

abs_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s\n' "$PWD/${1#./}" ;;
    esac
}

project_version() {
    awk -F"'" '/version:/ { print $2; exit }' "$ROOT_DIR/meson.build" 2>/dev/null || true
}

normalize_arch() {
    case "$1" in
        x86_64|amd64) printf 'x86_64\tx64\n' ;;
        aarch64|arm64) printf 'aarch64\tarm64\n' ;;
        *) die "unsupported Linux architecture: $1" ;;
    esac
}

resolve_packager() {
    if [[ -n "${PACKAGER:-}" ]]; then
        PACKAGER="$(abs_path "$PACKAGER")"
        [[ -x "$PACKAGER" ]] || die "packaging CLI is not executable: $PACKAGER"
        printf '%s\n' "$PACKAGER"
        return
    fi
    local tool_dir="$OUTPUT_DIR/.tools/vpk"
    local tool="$tool_dir/vpk"
    local version_dir="$tool_dir/.store/vpk/$PACKAGE_TOOL_VERSION"
    if [[ -x "$tool" && -d "$version_dir" ]]; then
        printf '%s\n' "$tool"
        return
    fi

    local dotnet="${DOTNET:-}"
    if [[ -z "$dotnet" ]] && command -v dotnet >/dev/null 2>&1; then
        dotnet="$(command -v dotnet)"
    fi
    if [[ -z "$dotnet" && -n "${DOTNET_ROOT:-}" && -x "$DOTNET_ROOT/dotnet" ]]; then
        dotnet="$DOTNET_ROOT/dotnet"
    fi
    [[ -x "$dotnet" ]] || die ".NET 8 SDK is required to install the packaging CLI"
    rm -rf "$tool_dir"
    mkdir -p "$tool_dir"
    "$dotnet" tool install --tool-path "$tool_dir" vpk --version "$PACKAGE_TOOL_VERSION" >&2
    [[ -x "$tool" && -d "$version_dir" ]] || die "packaging CLI $PACKAGE_TOOL_VERSION was not installed correctly"
    printf '%s\n' "$tool"
}

if [[ $# -ne 0 ]]; then
    die "create-package.sh takes no arguments; configure it with environment variables"
fi

[[ -n "$VERSION" ]] || VERSION="$(project_version)"
[[ -n "$VERSION" ]] || die "cannot determine project version"
[[ -n "$ARCH" ]] || ARCH="$(uname -m)"
read -r ARCH RUNTIME_ARCH <<< "$(normalize_arch "$ARCH")"
RUNTIME="linux-$RUNTIME_ARCH"
CHANNEL="linux-$RUNTIME_ARCH$APPIMAGE_SUFFIX"
[[ -n "$PACK_ID" ]] || PACK_ID="$APP_ID"

CYBERETHER_BINARY="$(abs_path "$CYBERETHER_BINARY")"
JETSTREAM_SO="$(abs_path "$JETSTREAM_SO")"
ICON_SOURCE="$(abs_path "$ICON_SOURCE")"
OUTPUT_DIR="$(abs_path "$OUTPUT_DIR")"
PACK_DIR="$(abs_path "${PACK_DIR:-$OUTPUT_DIR/.pack}")"
RELEASE_NOTES="$(abs_path "$RELEASE_NOTES")"
if [[ -n "${DOTNET:-}" ]]; then
    DOTNET="$(abs_path "$DOTNET")"
    [[ -x "$DOTNET" ]] || die ".NET executable does not exist: $DOTNET"
    if [[ -z "${DOTNET_ROOT:-}" ]]; then
        export DOTNET_ROOT="${DOTNET%/*}"
    fi
fi

for path in "$CYBERETHER_BINARY" "$JETSTREAM_SO" "$ICON_SOURCE"; do
    [[ -f "$path" ]] || die "packaging input does not exist: $path"
done

rm -rf "$PACK_DIR"
mkdir -p "$PACK_DIR"

cp "$CYBERETHER_BINARY" "$PACK_DIR/$EXECUTABLE_NAME"
cp "$JETSTREAM_SO" "$PACK_DIR/$JETSTREAM_SO_NAME"
chmod 755 "$PACK_DIR/$EXECUTABLE_NAME" "$PACK_DIR/$JETSTREAM_SO_NAME"

patchelf --set-soname "$JETSTREAM_SO_NAME" "$PACK_DIR/$JETSTREAM_SO_NAME"
load_path="$(readelf -d "$PACK_DIR/$EXECUTABLE_NAME" | awk -F'[][]' '/NEEDED/ && $2 ~ /libjetstream\.so/ { print $2; exit }')"
[[ -n "$load_path" ]] || die "$EXECUTABLE_NAME does not link to $JETSTREAM_SO_NAME"
if [[ "$load_path" != "$JETSTREAM_SO_NAME" ]]; then
    patchelf --replace-needed "$load_path" "$JETSTREAM_SO_NAME" "$PACK_DIR/$EXECUTABLE_NAME"
fi
patchelf --set-rpath '$ORIGIN' "$PACK_DIR/$EXECUTABLE_NAME"
patchelf --set-rpath '$ORIGIN' "$PACK_DIR/$JETSTREAM_SO_NAME"

PACKAGER_PATH="$(resolve_packager)"
"$PACKAGER_PATH" pack \
    --packId "$PACK_ID" \
    --packVersion "$VERSION" \
    --packDir "$PACK_DIR" \
    --mainExe "$EXECUTABLE_NAME" \
    --packTitle "$APP_NAME" \
    --packAuthors "Luigi Cruz" \
    --icon "$ICON_SOURCE" \
    --outputDir "$OUTPUT_DIR" \
    --channel "$CHANNEL" \
    --runtime "$RUNTIME" \
    --releaseNotes "$RELEASE_NOTES" \
    --categories "Science;Engineering"

printf 'Created Linux release in: %s\n' "$OUTPUT_DIR"
