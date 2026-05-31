# CyberEther for Linux

This directory is packaging only. It wraps an already-built Linux `cyberether` executable and `libjetstream.so` into an AppImage.

## Usage

Build the executable and shared library, then package them:

```sh
meson setup build --default-library=shared -Dtests=false -Dexamples=false -Dremote=enabled
meson compile -C build cyberether
apps/linux/create-appimage.sh
```

The default output is:

```text
.dist/linux/CyberEther-<version>-<arch>.AppImage
```

The AppImage contains `usr/bin/cyberether` and `usr/lib/libjetstream.so`. `cyberether` is patched to load `libjetstream.so` from the bundled `usr/lib` directory, and AppRun exports that directory through `LD_LIBRARY_PATH` so plugins can resolve the same library.

Use environment variables to override packaging inputs when needed:

```sh
export CYBERETHER_BINARY="build/cyberether"
export JETSTREAM_SO="build/libjetstream.so"
export OUTPUT_DIR=".dist/linux"
apps/linux/create-appimage.sh
```

`patchelf` and `readelf` are required. If `appimagetool` is not available on `PATH`, the script downloads the matching AppImageKit continuous build with `curl`.
