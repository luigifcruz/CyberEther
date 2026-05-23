#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
umask 077
PATH="/usr/bin:/bin:/usr/sbin:/sbin"

APP_NAME="${APP_NAME:-CyberEther}"
EXECUTABLE_NAME="${EXECUTABLE_NAME:-cyberether}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

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

require_env() {
    [[ -n "${!1:-}" ]] || die "$1 is required"
}

assert_authorized_release_context() {
    [[ "${GITHUB_ACTIONS:-}" == "true" ]] || return 0
    [[ "${GITHUB_EVENT_NAME:-}" == "push" ]] || die "signing is only allowed for GitHub tag push events"
    [[ "${GITHUB_REF_TYPE:-}" == "tag" ]] || die "signing is only allowed for GitHub tag refs"
    [[ "${GITHUB_REF_NAME:-}" == v* ]] || die "signing is only allowed for v* tags"
}

validate_metadata() {
    [[ "$APP_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._\ -]*$ ]] || die "APP_NAME contains unsupported characters"
    [[ "$EXECUTABLE_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "EXECUTABLE_NAME contains unsupported characters"
}

decode_base64_to_file() {
    local value="$1"
    local path="$2"

    if printf '%s' "$value" | base64 --decode > "$path" 2>/dev/null; then
        return 0
    fi

    printf '%s' "$value" | base64 -D > "$path"
}

if [[ $# -ne 0 ]]; then
    die "sign-notarize-app.sh takes no arguments; configure it with environment variables"
fi

OUTPUT_DIR="$(abs_path "$OUTPUT_DIR")"
APP_PATH="$(abs_path "${APP_PATH:-$OUTPUT_DIR/$APP_NAME.app}")"
APP_ZIP_PATH="$(abs_path "${APP_ZIP_PATH:-$OUTPUT_DIR/$APP_NAME.app.zip}")"
FRAMEWORKS_DIR="$APP_PATH/Contents/Frameworks"
JETSTREAM_DYLIB_PATH="$FRAMEWORKS_DIR/libjetstream.dylib"
ENTITLEMENTS_PATH="$SCRIPT_DIR/Entitlements.plist"
validate_metadata
assert_authorized_release_context

WORK_ROOT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
WORK_DIR="$(mktemp -d "$WORK_ROOT/cyberether-sign-app.XXXXXX")"
KEYCHAIN_PATH="$WORK_DIR/signing.keychain-db"
KEYCHAIN_PASSWORD="${KEYCHAIN_PASSWORD:-$(uuidgen)}"
CERT_PATH="$WORK_DIR/certificate.p12"
NOTARY_KEY_PATH="$WORK_DIR/AuthKey.p8"

cleanup() {
    security delete-keychain "$KEYCHAIN_PATH" >/dev/null 2>&1 || true
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

validate_inputs() {
    [[ -d "$APP_PATH" ]] || die "app bundle does not exist: $APP_PATH"
    [[ -f "$APP_PATH/Contents/MacOS/$EXECUTABLE_NAME" ]] || die "app executable does not exist: $APP_PATH/Contents/MacOS/$EXECUTABLE_NAME"
    [[ -f "$JETSTREAM_DYLIB_PATH" ]] || die "app dylib does not exist: $JETSTREAM_DYLIB_PATH"
    [[ -f "$ENTITLEMENTS_PATH" ]] || die "entitlements plist does not exist: $ENTITLEMENTS_PATH"

    require_env APPLE_CERT_P12_BASE64
    require_env APPLE_CERT_PASSWORD
    require_env APPLE_CODESIGN_IDENTITY
    require_env APPLE_NOTARY_KEY_ID
    require_env APPLE_NOTARY_ISSUER_ID
    require_env APPLE_NOTARY_KEY_P8_BASE64
}

setup_credentials() {
    require_tool base64
    require_tool security
    require_tool uuidgen

    decode_base64_to_file "$APPLE_CERT_P12_BASE64" "$CERT_PATH"
    decode_base64_to_file "$APPLE_NOTARY_KEY_P8_BASE64" "$NOTARY_KEY_PATH"
    chmod 600 "$CERT_PATH" "$NOTARY_KEY_PATH"

    security create-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH" >/dev/null
    security set-keychain-settings -lut 21600 "$KEYCHAIN_PATH" >/dev/null
    security unlock-keychain -p "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH" >/dev/null
    security import "$CERT_PATH" -k "$KEYCHAIN_PATH" -P "$APPLE_CERT_PASSWORD" -T /usr/bin/codesign -T /usr/bin/security >/dev/null
    security set-key-partition-list -S apple-tool:,apple:,codesign: -s -k "$KEYCHAIN_PASSWORD" "$KEYCHAIN_PATH" >/dev/null
    security list-keychains -d user -s "$KEYCHAIN_PATH" >/dev/null
}

sign_frameworks() {
    local dylib

    require_tool codesign

    for dylib in "$FRAMEWORKS_DIR"/*.dylib; do
        [[ -e "$dylib" ]] || continue
        codesign --force --options runtime --timestamp --sign "$APPLE_CODESIGN_IDENTITY" "$dylib"
        codesign --verify --strict --verbose=2 "$dylib"
    done
}

sign_app() {
    require_tool codesign

    codesign --force --options runtime --timestamp --entitlements "$ENTITLEMENTS_PATH" --sign "$APPLE_CODESIGN_IDENTITY" "$APP_PATH"
    codesign --verify --strict --verbose=2 "$APP_PATH"
}

notarize_app() {
    require_tool ditto
    require_tool spctl
    require_tool xcrun

    rm -f "$APP_ZIP_PATH"
    ditto -c -k --keepParent "$APP_PATH" "$APP_ZIP_PATH"

    xcrun notarytool submit "$APP_ZIP_PATH" \
        --key "$NOTARY_KEY_PATH" \
        --key-id "$APPLE_NOTARY_KEY_ID" \
        --issuer "$APPLE_NOTARY_ISSUER_ID" \
        --wait

    xcrun stapler staple "$APP_PATH"
    xcrun stapler validate "$APP_PATH"
    spctl --assess --type execute --verbose "$APP_PATH"

    rm -f "$APP_ZIP_PATH"
}

validate_inputs
setup_credentials
sign_frameworks
sign_app
notarize_app
msg "Signed and notarized app bundle: $APP_PATH"
