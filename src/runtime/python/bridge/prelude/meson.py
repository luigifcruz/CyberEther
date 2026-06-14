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


def generate_header(input_paths, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    components = []
    for input_path in input_paths:
        name = os.path.basename(input_path)
        body = read_component(input_path).rstrip("\n")
        components.append(f"# === [bridge/prelude/{name}] ===\n\n{body}\n")

    content = "\n\n".join(components)
    delimiter = raw_string_delimiter(content)

    with open(output_path, "w", encoding="utf-8") as output:
        output.write("#pragma once\n\n")
        output.write("namespace Jetstream {\n\n")
        output.write("inline constexpr const char* kPythonBridge = ")
        output.write(f'R"{delimiter}({content}){delimiter}"')
        output.write(";\n\n")
        output.write("}  // namespace Jetstream\n")


if __name__ == "__main__":
    generate_header(sys.argv[2:], sys.argv[1])
