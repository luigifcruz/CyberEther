#!/bin/python3

import os
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


def collect_inputs(inputs, stub):
    """Only package declared build inputs, never stale files from an old build."""
    shaders = {}
    kernels = {}
    for filepath in inputs:
        basename, stage, target = os.path.basename(filepath).rsplit(".", 2)
        if not basename.startswith(stub + "_") or target_to_device(target) is None:
            raise ValueError(f"Unexpected shader input: {filepath}")
        name = basename[len(stub) + 1:]
        if stage not in ("vert", "frag", "comp"):
            raise ValueError(f"Unexpected shader stage: {filepath}")
        entries = kernels if stage == "comp" else shaders
        stages = entries.setdefault(name, {}).setdefault(target, {})
        if stage in stages:
            raise ValueError(f"Duplicate shader input: {filepath}")
        stages[stage] = filepath
    for name, targets in shaders.items():
        for target, stages in targets.items():
            if set(stages) != {"vert", "frag"}:
                raise ValueError(f"Incomplete shader pair: {name}.{target}")
    return shaders, kernels


def bin_to_header(path, stub, inputs):
    shaders, kernels = collect_inputs(inputs, stub)
    path = os.path.join(path, "resources", "shaders")
    decorator = "Global" if stub == "global" else ""

    with open(os.path.join(path, f"{stub}_shaders.hh"), "w") as f:
        f.write("#pragma once\n\n")

        f.write("#include <vector>\n")
        f.write("#include <unordered_map>\n\n")

        f.write('#include "jetstream/memory/types.hh"\n\n')

        f.write("using namespace Jetstream;\n\n")

        # Package shaders.

        types = ("vert", "frag")

        for name, targets in sorted(shaders.items()):
            for target, stages in sorted(targets.items()):
                for stage in types:
                    varname = f"{name}_{target}_{stage}_shader"
                    with open(stages[stage], "rb") as fr:
                        data = fr.read()
                    f.write("static const std::vector<U8> " + varname + " = {\n    ")
                    f.write(format_hex_vector(data))
                    f.write("\n};\n\n")

        f.write(
            f"static std::unordered_map<std::string, std::unordered_map<Jetstream::DeviceType, "
            f"std::vector<std::vector<U8>>>> {decorator}ShadersPackage = {{\n"
        )

        for name, targets in sorted(shaders.items()):
            f.write("    {\n")
            f.write(f'        "{name}", {{\n')

            for target in sorted(targets):
                device_str = target_to_device(target)

                f.write(f"            {{ {device_str}, {{ ")
                for stage in types:
                    f.write(f"{name}_{target}_{stage}_shader, ")
                f.write("} },\n")

            f.write("        }\n")
            f.write("    },\n")

        f.write("};\n\n")

        # Package kernels.

        for name, targets in sorted(kernels.items()):
            for target, stages in sorted(targets.items()):
                varname = f"{name}_{target}_kernel"
                with open(stages["comp"], "rb") as fr:
                    data = fr.read()
                f.write("static const std::vector<U8> " + varname + " = {\n    ")
                f.write(format_hex_vector(data))
                f.write("\n};\n\n")

        f.write(
            f"static std::unordered_map<std::string, std::unordered_map<Jetstream::DeviceType, "
            f"std::vector<std::vector<U8>>>> {decorator}KernelsPackage = {{\n"
        )

        for name, targets in sorted(kernels.items()):
            f.write("    {\n")
            f.write(f'        "{name}", {{\n')

            for target in sorted(targets):
                device_str = target_to_device(target)

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
    bin_to_header(sys.argv[1], sys.argv[2], sys.argv[3:])
