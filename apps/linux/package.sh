#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: apps/linux/package.sh --binary PATH [--output DIR] [--version VERSION] [--arch ARCH]

Build unsigned Linux release packages for an existing cyberether binary.
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
    x86_64|amd64)
        arch='x86_64'
        deb_arch='amd64'
        appimage_arch='x86_64'
        ;;
    aarch64|arm64)
        arch='aarch64'
        deb_arch='arm64'
        appimage_arch='aarch64'
        ;;
    *)
        deb_arch="$arch"
        appimage_arch="$arch"
        ;;
esac

work_dir="$repo_root/.dist/linux"
appdir="$work_dir/CyberEther.AppDir"
metainfo="$work_dir/io.github.luigifcruz.CyberEther.metainfo.xml"
mkdir -p "$work_dir"
rm -rf "$appdir"

release_date="$(date -u +%F)"
sed \
    -e "s/@VERSION@/$version/g" \
    -e "s/@DATE@/$release_date/g" \
    "$repo_root/apps/linux/io.github.luigifcruz.CyberEther.metainfo.xml.in" > "$metainfo"

copy_payload() {
    local root="$1"

    mkdir -p \
        "$root/usr/bin" \
        "$root/usr/share/applications" \
        "$root/usr/share/metainfo" \
        "$root/usr/share/icons/hicolor/scalable/apps" \
        "$root/usr/share/licenses/cyberether"

    cp "$binary" "$root/usr/bin/cyberether"
    chmod 755 "$root/usr/bin/cyberether"
    cp "$repo_root/apps/linux/cyberether.desktop" "$root/usr/share/applications/cyberether.desktop"
    cp "$metainfo" "$root/usr/share/metainfo/io.github.luigifcruz.CyberEther.metainfo.xml"
    cp "$repo_root/resources/icons/cyberether.svg" "$root/usr/share/icons/hicolor/scalable/apps/cyberether.svg"
    cp "$repo_root/LICENSE" "$root/usr/share/licenses/cyberether/LICENSE"
}

copy_payload "$appdir"
cp "$repo_root/apps/linux/cyberether.desktop" "$appdir/cyberether.desktop"
cp "$repo_root/resources/icons/cyberether.svg" "$appdir/cyberether.svg"

cat > "$appdir/AppRun" <<'EOF'
#!/usr/bin/env sh
set -eu
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/cyberether" "$@"
EOF
chmod 755 "$appdir/AppRun"

tar -C "$work_dir" -czf "$output_dir/cyberether-linux-${arch}-AppDir.tar.gz" CyberEther.AppDir

appimagetool_bin="${APPIMAGETOOL:-}"
if [ -z "$appimagetool_bin" ] && command -v appimagetool >/dev/null 2>&1; then
    appimagetool_bin="$(command -v appimagetool)"
fi

if [ -n "$appimagetool_bin" ] && [ -x "$appimagetool_bin" ]; then
    ARCH="$appimage_arch" "$appimagetool_bin" "$appdir" "$output_dir/cyberether-linux-${arch}.AppImage"
else
    printf 'appimagetool not found; skipped AppImage creation.\n' >&2
fi

deb_root="$work_dir/deb/cyberether_${version}_${deb_arch}"
deb_tmp="$work_dir/deb-tmp"
rm -rf "$deb_root" "$deb_tmp"
mkdir -p "$deb_root/DEBIAN" "$deb_tmp"
copy_payload "$deb_root"
installed_size="$(du -sk "$deb_root/usr" | cut -f1)"

cat > "$deb_root/DEBIAN/control" <<EOF
Package: cyberether
Version: $version
Section: science
Priority: optional
Architecture: $deb_arch
Installed-Size: $installed_size
Maintainer: Luigi Cruz <hello@luigi.ltd>
Homepage: https://cyberether.org
Description: GPU-accelerated signal processing and visualization
 CyberEther is a native, multi-platform GPU-accelerated application for
 real-time signal visualization, software-defined radio workflows, and
 compute-intensive pipelines.
EOF

printf '2.0\n' > "$deb_tmp/debian-binary"
(cd "$deb_root/DEBIAN" && tar --owner=0 --group=0 -czf "$deb_tmp/control.tar.gz" .)
(cd "$deb_root" && tar --owner=0 --group=0 -czf "$deb_tmp/data.tar.gz" ./usr)
(cd "$deb_tmp" && ar rcs "$output_dir/cyberether-linux-${arch}.deb" debian-binary control.tar.gz data.tar.gz)

if command -v rpmbuild >/dev/null 2>&1; then
    rpm_top="$work_dir/rpmbuild"
    rm -rf "$rpm_top"
    mkdir -p "$rpm_top/BUILD" "$rpm_top/BUILDROOT" "$rpm_top/RPMS" "$rpm_top/SOURCES" "$rpm_top/SPECS" "$rpm_top/SRPMS"
    cp "$binary" "$rpm_top/SOURCES/cyberether"
    cp "$repo_root/apps/linux/cyberether.desktop" "$rpm_top/SOURCES/cyberether.desktop"
    cp "$metainfo" "$rpm_top/SOURCES/io.github.luigifcruz.CyberEther.metainfo.xml"
    cp "$repo_root/resources/icons/cyberether.svg" "$rpm_top/SOURCES/cyberether.svg"
    cp "$repo_root/LICENSE" "$rpm_top/SOURCES/LICENSE"

    cat > "$rpm_top/SPECS/cyberether.spec" <<'EOF'
Name: cyberether
Version: %{_version}
Release: 1%{?dist}
Summary: GPU-accelerated signal processing and visualization
License: MIT
URL: https://cyberether.org

%description
CyberEther is a native, multi-platform GPU-accelerated application for
real-time signal visualization, software-defined radio workflows, and
compute-intensive pipelines.

%prep

%build

%install
install -Dm755 %{_sourcedir}/cyberether %{buildroot}/usr/bin/cyberether
install -Dm644 %{_sourcedir}/cyberether.desktop %{buildroot}/usr/share/applications/cyberether.desktop
install -Dm644 %{_sourcedir}/io.github.luigifcruz.CyberEther.metainfo.xml %{buildroot}/usr/share/metainfo/io.github.luigifcruz.CyberEther.metainfo.xml
install -Dm644 %{_sourcedir}/cyberether.svg %{buildroot}/usr/share/icons/hicolor/scalable/apps/cyberether.svg
install -Dm644 %{_sourcedir}/LICENSE %{buildroot}/usr/share/licenses/cyberether/LICENSE

%files
%license /usr/share/licenses/cyberether/LICENSE
/usr/bin/cyberether
/usr/share/applications/cyberether.desktop
/usr/share/metainfo/io.github.luigifcruz.CyberEther.metainfo.xml
/usr/share/icons/hicolor/scalable/apps/cyberether.svg
EOF

    rpmbuild -bb "$rpm_top/SPECS/cyberether.spec" \
        --define "_topdir $rpm_top" \
        --define "_version $version" \
        --define '_build_id_links none'

    rpm_file="$(find "$rpm_top/RPMS" -type f -name '*.rpm' -print -quit)"
    if [ -n "$rpm_file" ]; then
        cp "$rpm_file" "$output_dir/cyberether-linux-${arch}.rpm"
    fi
else
    printf 'rpmbuild not found; skipped RPM creation.\n' >&2
fi
