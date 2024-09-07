#!/bin/bash

FMT_PATH="$(pwd)/../include/jetstream/tools/fmt"

# Determine the OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    SED_I=(-i "")
else
    # Linux and others
    SED_I=(-i)
fi

# Execute the find command
find "$FMT_PATH" \( -name '*.h' -o -name '*.cc' \) -exec sed "${SED_I[@]}" \
    's/fmt::/jst::fmt::/g; s/namespace fmt/namespace jst::fmt/g; s/FMT_/JST_FMT_/g;' '{}' \;