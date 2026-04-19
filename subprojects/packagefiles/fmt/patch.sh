#!/bin/bash

set -e

FMT_PATH="$1"
FMT_ROOT="$(dirname "$FMT_PATH")"
STAMP_PATH="$FMT_ROOT/patch_headers.stamp"
JETSTREAM_FMT_ROOT="$FMT_ROOT/jetstream/fmt"

if [ -f "$STAMP_PATH" ] && [ -f "$JETSTREAM_FMT_ROOT/format.h" ]; then
    exit 0
fi

# Determine the OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SED_I=(-i "")
else
    # Linux and others
    SED_I=(-i)
fi

if [ ! -f "$STAMP_PATH" ]; then
    find "$FMT_PATH" \( -name '*.h' -o -name '*.cc' \) -exec sed "${SED_I[@]}" \
        's/fmt::/jst::fmt::/g; s/namespace fmt/namespace jst::fmt/g; s/FMT_/JST_FMT_/g;' '{}' \;
fi

rm -rf "$JETSTREAM_FMT_ROOT"
mkdir -p "$(dirname "$JETSTREAM_FMT_ROOT")"
cp -R "$FMT_PATH" "$JETSTREAM_FMT_ROOT"

# Dummy file for Meson target
touch "$STAMP_PATH"
