#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
PATH="/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

APP_NAME="${APP_NAME:-CyberEther}"
PACK_ID="${PACK_ID:-CyberEther}"
BUNDLE_ID="${BUNDLE_ID:-ltd.luigi.CyberEther}"
EXECUTABLE_NAME="${EXECUTABLE_NAME:-cyberether}"
MIN_MACOS="${MIN_MACOS:-13.0}"
CHANNEL="${CHANNEL:-osx-arm64}"
RUNTIME="${RUNTIME:-osx-arm64}"
PACKAGE_TOOL_VERSION="${PACKAGE_TOOL_VERSION:-1.2.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION="${VERSION:-}"
CYBERETHER_BINARY="${CYBERETHER_BINARY:-$ROOT_DIR/build/cyberether}"
JETSTREAM_DYLIB="${JETSTREAM_DYLIB:-$ROOT_DIR/build/libjetstream.dylib}"
ICON_SOURCE="${ICON_SOURCE:-$ROOT_DIR/resources/assets/icon.png}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/.dist/macos}"
RELEASE_NOTES="${RELEASE_NOTES:-/dev/null}"

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
    [[ "$VERSION" =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || die "VERSION contains unsupported characters"
}

escape_sed() {
    printf '%s' "$1" | sed -e 's/[\/&|]/\\&/g'
}

project_version() {
    awk -F"'" '/version:/ { print $2; exit }' "$ROOT_DIR/meson.build" 2>/dev/null || true
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

decode_base64_to_file() {
    local value="$1"
    local path="$2"
    if printf '%s' "$value" | base64 --decode > "$path" 2>/dev/null; then
        return
    fi
    printf '%s' "$value" | base64 -D > "$path"
}

has_rpath() {
    otool -l "$1" | awk -v expected="$2" '
        $1 == "cmd" && $2 == "LC_RPATH" { in_rpath = 1; next }
        in_rpath && $1 == "path" {
            if ($2 == expected) found = 1
            in_rpath = 0
        }
        END { exit found ? 0 : 1 }
    '
}

fix_library_paths() {
    local app_binary="$PACK_DIR/$EXECUTABLE_NAME"
    local app_dylib="$PACK_DIR/$JETSTREAM_DYLIB_NAME"
    local load_path

    install_name_tool -id "@rpath/$JETSTREAM_DYLIB_NAME" "$app_dylib"

    load_path="$(otool -L "$app_binary" | awk '/libjetstream.*\.dylib/ { print $1; exit }')"
    [[ -n "$load_path" ]] || die "$EXECUTABLE_NAME does not link to $JETSTREAM_DYLIB_NAME"
    if [[ "$load_path" != "@rpath/$JETSTREAM_DYLIB_NAME" ]]; then
        install_name_tool -change "$load_path" "@rpath/$JETSTREAM_DYLIB_NAME" "$app_binary"
    fi

    if ! has_rpath "$app_binary" "@executable_path"; then
        install_name_tool -add_rpath "@executable_path" "$app_binary"
    fi
}

render_plist() {
    sed \
        -e "s|@APP_NAME@|$(escape_sed "$APP_NAME")|g" \
        -e "s|@EXECUTABLE_NAME@|$(escape_sed "$EXECUTABLE_NAME")|g" \
        -e "s|@BUNDLE_ID@|$(escape_sed "$BUNDLE_ID")|g" \
        -e "s|@VERSION@|$(escape_sed "$VERSION")|g" \
        -e "s|@MIN_MACOS@|$(escape_sed "$MIN_MACOS")|g" \
        "$INFO_PLIST_TEMPLATE" > "$PLIST_PATH"
    plutil -lint "$PLIST_PATH" >/dev/null
}

generate_icon() {
    local iconset="$WORK_DIR/AppIcon.iconset"
    local size

    mkdir -p "$iconset"
    for size in 16 32 128 256 512; do
        local double_size=$((size * 2))
        sips -z "$size" "$size" "$ICON_SOURCE" --out "$iconset/icon_${size}x${size}.png" >/dev/null
        sips -z "$double_size" "$double_size" "$ICON_SOURCE" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
    done
    iconutil -c icns "$iconset" -o "$ICON_PATH"
}

setup_signing() {
    local keychain_list

    require_tool base64
    require_tool codesign
    require_tool security
    require_tool spctl
    require_tool uuidgen
    require_tool xcrun

    keychain_list="$(security list-keychains -d user)"
    while IFS= read -r keychain; do
        keychain="${keychain#*\"}"
        keychain="${keychain%\"*}"
        [[ -n "$keychain" ]] && ORIGINAL_KEYCHAINS+=("$keychain")
    done <<< "$keychain_list"
    KEYCHAIN_SEARCH_LIST_CAPTURED=true

    SIGNING_WORK_DIR="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/cyberether-package.XXXXXX")"
    KEYCHAIN_PATH="$SIGNING_WORK_DIR/signing.keychain-db"
    KEYCHAIN_PASSWORD="${KEYCHAIN_PASSWORD:-$(uuidgen)}"
    APP_CERT_PATH="$SIGNING_WORK_DIR/application-certificate.p12"
    INSTALLER_CERT_PATH="$SIGNING_WORK_DIR/installer-certificate.p12"
    NOTARY_KEY_PATH="$SIGNING_WORK_DIR/AuthKey.p8"
    ENTITLEMENTS_PATH="$SIGNING_WORK_DIR/CyberEther.entitlements"

    decode_base64_to_file "$APPLE_APP_CERT_P12_BASE64" "$APP_CERT_PATH"
    decode_base64_to_file "$APPLE_INSTALLER_CERT_P12_BASE64" "$INSTALLER_CERT_PATH"
    decode_base64_to_file "$APPLE_NOTARY_KEY_P8_BASE64" "$NOTARY_KEY_PATH"
    cp "$ENTITLEMENTS_SOURCE" "$ENTITLEMENTS_PATH"
    chmod 600 "$APP_CERT_PATH" "$INSTALLER_CERT_PATH" "$NOTARY_KEY_PATH"

    security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH" >/dev/null
    security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH" >/dev/null
    security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH" >/dev/null
    security list-keychains -d user -s "$KEYCHAIN_PATH" "${ORIGINAL_KEYCHAINS[@]}"
    security import "$APP_CERT_PATH" -k "$KEYCHAIN_PATH" -P "$APPLE_APP_CERT_PASSWORD" -T /usr/bin/codesign -T /usr/bin/security >/dev/null
    security import "$INSTALLER_CERT_PATH" -k "$KEYCHAIN_PATH" -P "$APPLE_INSTALLER_CERT_PASSWORD" -T /usr/bin/productbuild -T /usr/bin/security >/dev/null
    security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH" >/dev/null
    xcrun notarytool store-credentials "$NOTARY_PROFILE" \
        --key "$NOTARY_KEY_PATH" \
        --key-id "$APPLE_NOTARY_KEY_ID" \
        --issuer "$APPLE_NOTARY_ISSUER_ID" \
        --keychain "$KEYCHAIN_PATH" >/dev/null
}

if [[ $# -ne 0 ]]; then
    die "create-package.sh takes no arguments; configure it with environment variables"
fi

[[ -n "$VERSION" ]] || VERSION="$(project_version)"
[[ -n "$VERSION" ]] || die "cannot determine project version"
validate_metadata

CYBERETHER_BINARY="$(abs_path "$CYBERETHER_BINARY")"
JETSTREAM_DYLIB="$(abs_path "$JETSTREAM_DYLIB")"
ICON_SOURCE="$(abs_path "$ICON_SOURCE")"
OUTPUT_DIR="$(abs_path "$OUTPUT_DIR")"
RELEASE_NOTES="$(abs_path "$RELEASE_NOTES")"
if [[ -n "${DOTNET:-}" ]]; then
    DOTNET="$(abs_path "$DOTNET")"
    [[ -x "$DOTNET" ]] || die ".NET executable does not exist: $DOTNET"
    if [[ -z "${DOTNET_ROOT:-}" ]]; then
        export DOTNET_ROOT="${DOTNET%/*}"
    fi
fi

PACK_DIR="$OUTPUT_DIR/.pack"
WORK_DIR="$OUTPUT_DIR/.work"
PLIST_PATH="$WORK_DIR/Info.plist"
ICON_PATH="$WORK_DIR/AppIcon.icns"
PKG_PATH="$OUTPUT_DIR/$PACK_ID-$CHANNEL-Setup.pkg"
JETSTREAM_DYLIB_NAME="libjetstream.dylib"
INFO_PLIST_TEMPLATE="$SCRIPT_DIR/Info.plist.in"
ENTITLEMENTS_SOURCE="$SCRIPT_DIR/Entitlements.plist"
NOTARY_PROFILE="cyberether-package"

SIGNING_WORK_DIR=""
KEYCHAIN_PATH=""
ORIGINAL_KEYCHAINS=()
KEYCHAIN_SEARCH_LIST_CAPTURED=false
cleanup() {
    if [[ -n "$KEYCHAIN_PATH" ]]; then
        if [[ "$KEYCHAIN_SEARCH_LIST_CAPTURED" == true ]]; then
            security list-keychains -d user -s "${ORIGINAL_KEYCHAINS[@]}" >/dev/null 2>&1 || true
        fi
        security delete-keychain "$KEYCHAIN_PATH" >/dev/null 2>&1 || true
    fi
    if [[ -n "$SIGNING_WORK_DIR" ]]; then
        rm -rf "$SIGNING_WORK_DIR"
    fi
}
trap cleanup EXIT

[[ -f "$CYBERETHER_BINARY" ]] || die "binary does not exist: $CYBERETHER_BINARY"
[[ -f "$JETSTREAM_DYLIB" ]] || die "dylib does not exist: $JETSTREAM_DYLIB"
[[ -f "$ICON_SOURCE" ]] || die "icon source does not exist: $ICON_SOURCE"
[[ -f "$INFO_PLIST_TEMPLATE" ]] || die "Info.plist template does not exist: $INFO_PLIST_TEMPLATE"
[[ -f "$ENTITLEMENTS_SOURCE" ]] || die "entitlements file does not exist: $ENTITLEMENTS_SOURCE"

for tool in iconutil install_name_tool otool pkgbuild plutil productbuild sed sips; do
    require_tool "$tool"
done

rm -rf "$PACK_DIR" "$WORK_DIR"
mkdir -p "$PACK_DIR" "$WORK_DIR"
cp "$CYBERETHER_BINARY" "$PACK_DIR/$EXECUTABLE_NAME"
cp "$JETSTREAM_DYLIB" "$PACK_DIR/$JETSTREAM_DYLIB_NAME"
chmod 755 "$PACK_DIR/$EXECUTABLE_NAME" "$PACK_DIR/$JETSTREAM_DYLIB_NAME"
fix_library_paths
render_plist
generate_icon

signing_available=false
if [[ -n "${APPLE_APP_CERT_P12_BASE64:-}" &&
      -n "${APPLE_APP_CERT_PASSWORD:-}" &&
      -n "${APPLE_CODESIGN_IDENTITY:-}" &&
      -n "${APPLE_INSTALLER_CERT_P12_BASE64:-}" &&
      -n "${APPLE_INSTALLER_CERT_PASSWORD:-}" &&
      -n "${APPLE_INSTALLER_IDENTITY:-}" &&
      -n "${APPLE_NOTARY_KEY_ID:-}" &&
      -n "${APPLE_NOTARY_ISSUER_ID:-}" &&
      -n "${APPLE_NOTARY_KEY_P8_BASE64:-}" ]]; then
    signing_available=true
fi
if [[ "${REQUIRE_SIGNING:-0}" == 1 && "$signing_available" != true ]]; then
    die "macOS application, installer, and notarization credentials are required"
fi
if [[ "$signing_available" == true ]]; then
    setup_signing
fi

PACKAGER_PATH="$(resolve_packager)"
PACK_ARGS=(
    pack
    --packId "$PACK_ID"
    --packVersion "$VERSION"
    --packDir "$PACK_DIR"
    --mainExe "$EXECUTABLE_NAME"
    --packTitle "$APP_NAME"
    --packAuthors "Luigi Cruz"
    --outputDir "$OUTPUT_DIR"
    --channel "$CHANNEL"
    --runtime "$RUNTIME"
    --icon "$ICON_PATH"
    --plist "$PLIST_PATH"
    --releaseNotes "$RELEASE_NOTES"
    --noPortable
)
if [[ "$signing_available" == true ]]; then
    PACK_ARGS+=(
        --signAppIdentity "$APPLE_CODESIGN_IDENTITY"
        --signInstallIdentity "$APPLE_INSTALLER_IDENTITY"
        --signEntitlements "$ENTITLEMENTS_PATH"
        --notaryProfile "$NOTARY_PROFILE"
        --keychain "$KEYCHAIN_PATH"
    )
fi

"$PACKAGER_PATH" "${PACK_ARGS[@]}"
[[ -f "$PKG_PATH" ]] || die "installer package was not created: $PKG_PATH"

msg "Created macOS release in: $OUTPUT_DIR"
