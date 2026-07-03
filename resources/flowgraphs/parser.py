#!/bin/python3

import os
import sys


def cpp_string(value):
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def yml_to_source(path, output, inputs):
    output_path = os.path.join(path, output)
    if not os.path.isdir(os.path.dirname(output_path)):
        output_path = output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write('#include "jetstream/registry.hh"\n\n')

        for file in inputs:
            filename = os.path.basename(file).split(".")[0]

            with open(os.path.join(path, file), "r", encoding="utf-8") as yml_file:
                content = yml_file.read()

            f.write(
                "JST_REGISTER_EXAMPLE("
                f'"{cpp_string(filename)}", '
                f'"{cpp_string(content)}");\n'
            )


if __name__ == "__main__":
    yml_to_source(sys.argv[1], sys.argv[2], sys.argv[3:])
