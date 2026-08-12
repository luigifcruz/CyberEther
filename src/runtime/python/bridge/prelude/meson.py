#!/bin/python3

import os
import sys


def raw_string_delimiter(content):
    base = "JST_PY_BRIDGE"
    delimiter = base
    index = 0
    while f'){delimiter}"' in content:
        index += 1
        delimiter = f"{base}_{index}"
    return delimiter


def read_component(input_path):
    with open(input_path, "r", encoding="utf-8") as source:
        return source.read()


# MSVC (C2026) rejects a single string literal larger than 16380 bytes. Split the
# content into smaller chunks emitted as adjacent raw string literals, which the
# compiler concatenates back together. Kept well under the limit for UTF-8 slack.
CHUNK_SIZE = 8000


def chunk_bytes(content):
    # Split on character boundaries with a byte budget so each emitted chunk is
    # valid UTF-8 on its own; concatenating them reproduces the original exactly.
    chunk = []
    size = 0
    for ch in content:
        ch_len = len(ch.encode("utf-8"))
        if size + ch_len > CHUNK_SIZE and chunk:
            yield "".join(chunk)
            chunk = []
            size = 0
        chunk.append(ch)
        size += ch_len
    if chunk:
        yield "".join(chunk)


def generate_header(input_paths, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    components = []
    for input_path in input_paths:
        name = os.path.basename(input_path)
        body = read_component(input_path).rstrip("\n")
        components.append(f"# === [bridge/prelude/{name}] ===\n\n{body}\n")

    content = "\n\n".join(components)
    delimiter = raw_string_delimiter(content)

    chunks = list(chunk_bytes(content)) or [""]

    with open(output_path, "w", encoding="utf-8") as output:
        output.write("#pragma once\n\n")
        output.write("namespace Jetstream {\n\n")
        output.write("inline constexpr const char* kPythonBridge =\n")
        for chunk in chunks:
            output.write(f'R"{delimiter}({chunk}){delimiter}"\n')
        output.write(";\n\n")
        output.write("}  // namespace Jetstream\n")


if __name__ == "__main__":
    generate_header(sys.argv[2:], sys.argv[1])
