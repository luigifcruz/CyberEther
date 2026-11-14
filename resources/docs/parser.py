#!/bin/python3

import os
import sys


def emit_bytes(f, name, data):
    f.write(f"const unsigned char {name}[] = {{\n")
    for start in range(0, len(data), 24):
        chunk = data[start:start + 24]
        f.write("    " + ", ".join(f"0x{byte:02x}" for byte in chunk) + ",\n")
    f.write("};\n\n")


def md_to_source(output, inputs):
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    entries = []

    with open(output, "w", encoding="utf-8") as f:
        f.write("#include <string_view>\n")
        f.write("#include <utility>\n")
        f.write("#include <vector>\n\n")
        f.write("namespace Jetstream::Resources {\n\n")
        f.write("namespace {\n\n")

        for index, path in enumerate(inputs):
            slug = os.path.splitext(os.path.basename(path))[0]
            with open(path, "rb") as source:
                data = source.read()
            name = f"kDoc{index}"
            emit_bytes(f, name, data)
            entries.append((slug, name, len(data)))

        f.write("}  // namespace\n\n")
        f.write("const std::vector<std::pair<std::string_view, std::string_view>>& BundledDocs() {\n")
        f.write("    static const std::vector<std::pair<std::string_view, std::string_view>> docs = {\n")
        for slug, name, size in entries:
            f.write(f'        {{"{slug}", std::string_view(reinterpret_cast<const char*>({name}), {size})}},\n')
        f.write("    };\n")
        f.write("    return docs;\n")
        f.write("}\n\n")
        f.write("}  // namespace Jetstream::Resources\n")


if __name__ == "__main__":
    md_to_source(sys.argv[1], sys.argv[2:])
