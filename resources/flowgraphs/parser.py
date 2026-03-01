#!/bin/python3

import glob
import os
import sys

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def yml_to_header(path, output, inputs):
    with open(os.path.join(path, output), "w") as f:
        f.write("#pragma once\n\n")

        f.write("#include <string>\n")
        f.write("#include <unordered_map>\n\n")

        f.write('#include "jetstream/registry.hh"\n\n')

        f.write("namespace Jetstream::Resources {\n\n")

        f.write(
            "inline Result GetDefaultManifest(std::vector<Registry::FlowgraphRegistration>& manifest) {\n"
        )

        for file in inputs:
            filename = os.path.basename(file).split(".")[0]

            with open(os.path.join(path, file), "r") as yml_file:
                if YAML_AVAILABLE:
                    yml_content = yaml.safe_load(yml_file)
                    title = yml_content.get("title", filename)
                    summary = yml_content.get("summary", "No summary.")
                    description = yml_content.get("description", "No description.")
                else:
                    title = filename
                    summary = "No summary."
                    description = "No description."

                yml_file.seek(0)
                content = yml_file.read().replace("\n", "\\n").replace('"', '\\"')

                f.write(
                    f'    manifest.push_back({{"{filename}", "{title}", "{summary}", "{description}", "{content}"}});\n'
                )

        f.write("    return Result::SUCCESS;\n")

        f.write("}\n\n")

        f.write("}  // namespace Jetstream::Resources\n")


if __name__ == "__main__":
    yml_to_header(sys.argv[1], sys.argv[2], sys.argv[3:])
