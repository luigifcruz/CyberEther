#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: apps/macos/package.sh --binary PATH [--output DIR] [--version VERSION] [--arch ARCH]

Build an unsigned CyberEther.app bundle and DMG from an existing cyberether binary.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"

binary=''
output_dir="$repo_root/.dist/artifacts"
version=''
arch="$(uname -m)"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --binary)
            binary="${2:?--binary requires a path}"
            shift 2
            ;;
        --output)
            output_dir="${2:?--output requires a directory}"
            shift 2
            ;;
        --version)
            version="${2:?--version requires a value}"
            shift 2
            ;;
        --arch)
            arch="${2:?--arch requires a value}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ -z "$binary" ]; then
    printf 'Missing required --binary argument.\n' >&2
    usage >&2
    exit 2
fi

if [ ! -f "$binary" ]; then
    printf 'Binary does not exist: %s\n' "$binary" >&2
    exit 1
fi

binary="$(cd "$(dirname "$binary")" && pwd)/$(basename "$binary")"
mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

project_version() {
    sed -n "s/.*version: '\([^']*\)'.*/\1/p" "$repo_root/meson.build" | sed -n '1p'
}

if [ -z "$version" ]; then
    version="$(project_version)"
fi

version="${version#v}"
if [ -z "$version" ]; then
    version='0.0.0'
fi

case "$arch" in
    arm64|aarch64)
        arch='arm64'
        ;;
    x86_64|amd64)
        arch='x86_64'
        ;;
esac

work_dir="$repo_root/.dist/macos"
app_dir="$work_dir/CyberEther.app"
contents_dir="$app_dir/Contents"
macos_dir="$contents_dir/MacOS"
resources_dir="$contents_dir/Resources"
rm -rf "$app_dir"
mkdir -p "$macos_dir" "$resources_dir"

cp "$binary" "$macos_dir/CyberEther"
chmod 755 "$macos_dir/CyberEther"
cp "$repo_root/LICENSE" "$resources_dir/LICENSE.txt"

icon_file=''
if [ -f "$repo_root/resources/icons/cyberether.icns" ]; then
    cp "$repo_root/resources/icons/cyberether.icns" "$resources_dir/cyberether.icns"
    icon_file='cyberether'
fi

sed \
    -e "s/@VERSION@/$version/g" \
    -e "s/@ICON_FILE@/$icon_file/g" \
    "$repo_root/apps/macos/Info.plist.in" > "$contents_dir/Info.plist"

tar -C "$work_dir" -czf "$output_dir/cyberether-macos-${arch}-app.tar.gz" CyberEther.app

if command -v hdiutil >/dev/null 2>&1; then
    hdiutil create \
        -volname CyberEther \
        -srcfolder "$app_dir" \
        -ov \
        -format UDZO \
        "$output_dir/cyberether-macos-${arch}.dmg"
else
    printf 'hdiutil not found; skipped DMG creation.\n' >&2
fi
