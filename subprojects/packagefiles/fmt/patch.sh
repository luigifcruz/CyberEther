#!/bin/bash

set -e

FMT_PATH="$1"

# Check if stamp exists
if [ -f "$(dirname "$FMT_PATH")/patch_headers.stamp" ]; then
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

find "$FMT_PATH" \( -name '*.h' -o -name '*.cc' \) -exec sed "${SED_I[@]}" \
    's/fmt::/jst::fmt::/g; s/namespace fmt/namespace jst::fmt/g; s/FMT_/JST_FMT_/g;' '{}' \;

# Dummy file for Meson target
touch "$(dirname "$FMT_PATH")/patch_headers.stamp"
