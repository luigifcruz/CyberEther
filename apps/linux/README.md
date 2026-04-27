# Linux Packaging

Linux release packaging for the native CyberEther desktop executable.

`package.sh` stages an existing `cyberether` binary and produces unsigned packages:

- `.deb` using a minimal Debian archive layout.
- `.rpm` when `rpmbuild` is available.
- `.AppImage` when `appimagetool` or `APPIMAGETOOL` is available.
- `AppDir.tar.gz` as a portable fallback and AppImage staging artifact.

This intentionally does not implement in-app updates or package signing.
