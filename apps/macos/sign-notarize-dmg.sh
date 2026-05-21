#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
umask 077
PATH="/usr/bin:/bin:/usr/sbin:/sbin"

APP_NAME="${APP_NAME:-CyberEther}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VERSION="${VERSION:-}"
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
}

decode_base64_to_file() {
    local value="$1"
    local path="$2"

    if printf '%s' "$value" | base64 --decode > "$path" 2>/dev/null; then
        return 0
    fi

    printf '%s' "$value" | base64 -D > "$path"
}

project_version() {
    awk -F"'" '/version:/ { print $2; exit }' "$ROOT_DIR/meson.build" 2>/dev/null || true
}

if [[ $# -ne 0 ]]; then
    die "sign-notarize-dmg.sh takes no arguments; configure it with environment variables"
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
DMG_PATH="$(abs_path "${DMG_PATH:-$OUTPUT_DIR/$APP_NAME-$VERSION.dmg}")"
assert_authorized_release_context

WORK_ROOT="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
WORK_DIR="$(mktemp -d "$WORK_ROOT/cyberether-sign-dmg.XXXXXX")"
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
    [[ -f "$DMG_PATH" ]] || die "DMG does not exist: $DMG_PATH"

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

sign_dmg() {
    require_tool codesign

    codesign --force --timestamp --sign "$APPLE_CODESIGN_IDENTITY" "$DMG_PATH"
    codesign --verify --verbose=2 "$DMG_PATH"
}

notarize_dmg() {
    require_tool spctl
    require_tool xcrun

    xcrun notarytool submit "$DMG_PATH" \
        --key "$NOTARY_KEY_PATH" \
        --key-id "$APPLE_NOTARY_KEY_ID" \
        --issuer "$APPLE_NOTARY_ISSUER_ID" \
        --wait

    xcrun stapler staple "$DMG_PATH"
    xcrun stapler validate "$DMG_PATH"
    spctl --assess --type open --context context:primary-signature --verbose "$DMG_PATH"
}

validate_inputs
setup_credentials
sign_dmg
notarize_dmg
msg "Signed and notarized installer DMG: $DMG_PATH"
