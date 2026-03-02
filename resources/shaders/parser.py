#!/bin/python3

import os
import glob
import sys


def format_hex_vector(data, bytes_per_line=16):
    values = [f"0x{byte:02x}" for byte in data]
    values.append("0x00")

    lines = []
    for i in range(0, len(values), bytes_per_line):
        lines.append(", ".join(values[i : i + bytes_per_line]))

    return ",\n    ".join(lines)


def target_to_device(target):
    if target == "msl":
        return "Jetstream::DeviceType::Metal"
    if target == "spv":
        return "Jetstream::DeviceType::Vulkan"
    if target == "wgsl":
        return "Jetstream::DeviceType::WebGPU"
    return None


def bin_to_header(path, stub):
    path = os.path.join(path, "resources", "shaders")
    decorator = "Global" if stub == "global" else ""

    with open(os.path.join(path, f"{stub}_shaders.hh"), "w") as f:
        f.write("#pragma once\n\n")

        f.write("#include <vector>\n")
        f.write("#include <unordered_map>\n\n")

        f.write('#include "jetstream/memory/types.hh"\n\n')

        f.write("using namespace Jetstream;\n\n")

        # Package shaders.

        targets = set()
        names = set()
        types = ("vert", "frag")

        for file in glob.glob(os.path.join(path, f"{stub}_*.vert.*")):
            target = file.split(".")[-1]
            targets.add(target)

            name = os.path.basename(file).split(".")[0].split("_")[1]
            names.add(name)

            for type in types:
                varname = f"{name}_{target}_{type}_shader"

                with open(
                    os.path.join(path, f"{stub}_{name}.{type}.{target}"), "rb"
                ) as fr:
                    data = fr.read()

                f.write("static const std::vector<U8> " + varname + " = {\n    ")
                f.write(format_hex_vector(data))
                f.write("\n};\n\n")

        f.write(
            f"static std::unordered_map<std::string, std::unordered_map<Jetstream::DeviceType, "
            f"std::vector<std::vector<U8>>>> {decorator}ShadersPackage = {{\n"
        )

        for name in sorted(names):
            f.write("    {\n")
            f.write(f'        "{name}", {{\n')

            for target in sorted(targets):
                device_str = target_to_device(target)
                if device_str is None:
                    continue

                f.write(f"            {{ {device_str}, {{ ")
                for type in types:
                    f.write(f"{name}_{target}_{type}_shader, ")
                f.write("} },\n")

            f.write("        }\n")
            f.write("    },\n")

        f.write("};\n\n")

        # Package kernels.

        targets = set()
        names = set()

        for file in glob.glob(os.path.join(path, f"{stub}_*.comp.*")):
            target = file.split(".")[-1]
            targets.add(target)

            name = os.path.basename(file).split(".")[0].split("_")[1]
            names.add(name)

            varname = f"{name}_{target}_kernel"

            with open(os.path.join(path, f"{stub}_{name}.comp.{target}"), "rb") as fr:
                data = fr.read()

            f.write("static const std::vector<U8> " + varname + " = {\n    ")
            f.write(format_hex_vector(data))
            f.write("\n};\n\n")

        f.write(
            f"static std::unordered_map<std::string, std::unordered_map<Jetstream::DeviceType, "
            f"std::vector<std::vector<U8>>>> {decorator}KernelsPackage = {{\n"
        )

        for name in sorted(names):
            f.write("    {\n")
            f.write(f'        "{name}", {{\n')

            for target in sorted(targets):
                device_str = target_to_device(target)
                if device_str is None:
                    continue

                f.write(
                    "            { "
                    + device_str
                    + ", { "
                    + f"{name}_{target}_kernel, "
                    + "} },\n"
                )

            f.write("        }\n")
            f.write("    },\n")

        f.write("};\n")


if __name__ == "__main__":
    bin_to_header(sys.argv[1], sys.argv[2])
