#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

APP_NAME="${APP_NAME:-CyberEther}"
APP_ID="${APP_ID:-ltd.luigi.CyberEther}"
EXECUTABLE_NAME="${EXECUTABLE_NAME:-cyberether}"
ICON_NAME="${ICON_NAME:-cyberether}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION="${VERSION:-}"
ARCH="${ARCH:-}"
CYBERETHER_BINARY="${CYBERETHER_BINARY:-$ROOT_DIR/build/cyberether}"
JETSTREAM_SO="${JETSTREAM_SO:-$ROOT_DIR/build/libjetstream.so}"
ICON_SOURCE="${ICON_SOURCE:-$ROOT_DIR/resources/assets/icon.png}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/.dist/linux}"
APPIMAGETOOL="${APPIMAGETOOL:-}"
APPIMAGETOOL_URL="${APPIMAGETOOL_URL:-}"
JETSTREAM_SO_NAME="libjetstream.so"

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

resolve_tool() {
    local tool="$1"

    if printf '%s' "$tool" | grep -q '/'; then
        tool="$(abs_path "$tool")"
        [[ -x "$tool" ]] || die "tool is not executable: $tool"
        printf '%s\n' "$tool"
        return 0
    fi

    command -v "$tool" >/dev/null 2>&1 || die "$tool is required"
    command -v "$tool"
}

download_appimagetool() {
    local appimagetool_path="$OUTPUT_DIR/.tools/appimagetool-$ARCH.AppImage"
    local appimagetool_url="${APPIMAGETOOL_URL:-https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-$ARCH.AppImage}"

    require_tool curl
    mkdir -p "${appimagetool_path%/*}"
    if [[ ! -x "$appimagetool_path" ]]; then
        msg "Downloading appimagetool: $appimagetool_url" >&2
        curl --fail --location --retry 3 --output "$appimagetool_path" "$appimagetool_url"
        chmod 755 "$appimagetool_path"
    fi

    printf '%s\n' "$appimagetool_path"
}

resolve_appimagetool() {
    if [[ -n "$APPIMAGETOOL" ]]; then
        resolve_tool "$APPIMAGETOOL"
        return 0
    fi

    if command -v appimagetool >/dev/null 2>&1; then
        command -v appimagetool
        return 0
    fi

    download_appimagetool
}

project_version() {
    awk -F"'" '/version:/ { print $2; exit }' "$ROOT_DIR/meson.build" 2>/dev/null || true
}

normalize_arch() {
    local machine="$1"

    case "$machine" in
        x86_64|amd64) printf 'x86_64\n' ;;
        aarch64|arm64) printf 'aarch64\n' ;;
        *) die "unsupported AppImage architecture: $machine" ;;
    esac
}

validate_metadata() {
    [[ "$APP_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._\ -]*$ ]] || die "APP_NAME contains unsupported characters"
    [[ "$APP_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*[A-Za-z0-9]$ ]] || die "APP_ID contains unsupported characters"
    [[ "$EXECUTABLE_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "EXECUTABLE_NAME contains unsupported characters"
    [[ "$ICON_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "ICON_NAME contains unsupported characters"
}

validate_inputs() {
    [[ -f "$CYBERETHER_BINARY" ]] || die "binary does not exist: $CYBERETHER_BINARY"
    [[ -f "$JETSTREAM_SO" ]] || die "shared library does not exist: $JETSTREAM_SO"
    [[ -f "$ICON_SOURCE" ]] || die "icon source does not exist: $ICON_SOURCE"
}

fix_library_paths() {
    local app_binary="$APPDIR_PATH/usr/bin/$EXECUTABLE_NAME"
    local app_lib="$APPDIR_PATH/usr/lib/$JETSTREAM_SO_NAME"
    local load_path

    require_tool patchelf
    require_tool readelf

    patchelf --set-soname "$JETSTREAM_SO_NAME" "$app_lib"

    load_path="$(readelf -d "$app_binary" | awk -F'[][]' '/NEEDED/ && $2 ~ /libjetstream\.so/ { print $2; exit }')"
    [[ -n "$load_path" ]] || die "$EXECUTABLE_NAME does not link to $JETSTREAM_SO_NAME"
    if [[ "$load_path" != "$JETSTREAM_SO_NAME" ]]; then
        patchelf --replace-needed "$load_path" "$JETSTREAM_SO_NAME" "$app_binary"
    fi

    patchelf --set-rpath '$ORIGIN/../lib' "$app_binary"
    patchelf --set-rpath '$ORIGIN' "$app_lib"
}

write_desktop_file() {
    local desktop_file="$APPDIR_PATH/$APP_ID.desktop"

    cat > "$desktop_file" <<DESKTOP
[Desktop Entry]
Type=Application
Name=$APP_NAME
Comment=CyberEther signal processing runtime
Exec=$EXECUTABLE_NAME %F
Icon=$ICON_NAME
Terminal=false
Categories=Science;Engineering;
DESKTOP

    cp "$desktop_file" "$APPDIR_PATH/usr/share/applications/$APP_ID.desktop"
}

write_apprun() {
    cat > "$APPDIR_PATH/AppRun" <<APPRUN
#!/bin/sh
set -eu
HERE="\${APPDIR:-\$(dirname "\$(readlink -f "\$0")")}"
export LD_LIBRARY_PATH="\$HERE/usr/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "\$HERE/usr/bin/$EXECUTABLE_NAME" "\$@"
APPRUN

    chmod 755 "$APPDIR_PATH/AppRun"
}

create_appdir() {
    rm -rf "$APPDIR_PATH"
    mkdir -p \
        "$APPDIR_PATH/usr/bin" \
        "$APPDIR_PATH/usr/lib" \
        "$APPDIR_PATH/usr/share/applications" \
        "$APPDIR_PATH/usr/share/icons/hicolor/256x256/apps"

    cp "$CYBERETHER_BINARY" "$APPDIR_PATH/usr/bin/$EXECUTABLE_NAME"
    cp "$JETSTREAM_SO" "$APPDIR_PATH/usr/lib/$JETSTREAM_SO_NAME"
    cp "$ICON_SOURCE" "$APPDIR_PATH/$ICON_NAME.png"
    cp "$ICON_SOURCE" "$APPDIR_PATH/.DirIcon"
    cp "$ICON_SOURCE" "$APPDIR_PATH/usr/share/icons/hicolor/256x256/apps/$ICON_NAME.png"

    chmod 755 "$APPDIR_PATH/usr/bin/$EXECUTABLE_NAME"
    chmod 755 "$APPDIR_PATH/usr/lib/$JETSTREAM_SO_NAME"

    fix_library_paths
    write_desktop_file
    write_apprun

    msg "Created AppDir: $APPDIR_PATH"
}

create_appimage() {
    local appimagetool_path

    appimagetool_path="$(resolve_appimagetool)"

    rm -f "$APPIMAGE_PATH"
    mkdir -p "${APPIMAGE_PATH%/*}"
    APPIMAGE_EXTRACT_AND_RUN="${APPIMAGE_EXTRACT_AND_RUN:-1}" \
        ARCH="$ARCH" \
        "$appimagetool_path" "$APPDIR_PATH" "$APPIMAGE_PATH"
    chmod 755 "$APPIMAGE_PATH"
    [[ -x "$APPIMAGE_PATH" ]] || die "AppImage was not created: $APPIMAGE_PATH"

    msg "Created AppImage: $APPIMAGE_PATH"
}

if [[ $# -ne 0 ]]; then
    die "create-appimage.sh takes no arguments; configure it with environment variables"
fi

if [[ -z "$VERSION" ]]; then
    VERSION="$(project_version)"
fi

if [[ -z "$VERSION" ]]; then
    die "cannot determine project version"
fi

if [[ -z "$ARCH" ]]; then
    ARCH="$(normalize_arch "$(uname -m)")"
else
    ARCH="$(normalize_arch "$ARCH")"
fi

[[ "$VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || die "VERSION contains unsupported characters"
validate_metadata

CYBERETHER_BINARY="$(abs_path "$CYBERETHER_BINARY")"
JETSTREAM_SO="$(abs_path "$JETSTREAM_SO")"
ICON_SOURCE="$(abs_path "$ICON_SOURCE")"
OUTPUT_DIR="$(abs_path "$OUTPUT_DIR")"
APPDIR_PATH="$(abs_path "${APPDIR_PATH:-$OUTPUT_DIR/$APP_NAME.AppDir}")"
APPIMAGE_PATH="$(abs_path "${APPIMAGE_PATH:-$OUTPUT_DIR/$APP_NAME-$VERSION-$ARCH.AppImage}")"

validate_inputs
mkdir -p "$OUTPUT_DIR"
create_appdir
create_appimage
