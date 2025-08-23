#!/bin/python3

import os
import glob
import sys

def bin_to_header(path, stub):
    path = os.path.join(path, 'resources', 'shaders')
    decorator = "Global" if stub == 'global' else ""

    with open(os.path.join(path, f'{stub}_shaders.hh'), 'w') as f:
        f.write('#pragma once\n\n')

        f.write('#include <vector>\n')
        f.write('#include <unordered_map>\n\n')

        f.write('#include "jetstream/memory/types.hh"\n\n')

        f.write('using namespace Jetstream;\n\n')

        # Package shaders.

        targets = set()
        names = set()
        types = ('vert', 'frag')

        for file in glob.glob(os.path.join(path, f'{stub}_*.vert.*')):
            target = file.split('.')[-1]
            targets.add(target)

            name = os.path.basename(file).split('.')[0].split('_')[1]
            names.add(name)

            for type in types:
                varname = f'{name}_{target}_{type}_shader'

                with open(os.path.join(path, f'{stub}_{name}.{type}.{target}'), 'rb') as fr:
                    data = fr.read()

                f.write('static const std::vector<U8> ' + varname + ' = {')
                f.write(','.join(f'0x{byte:02x}' for byte in data))
                f.write(',0x00};\n')

        f.write('\n')
        f.write(f'static std::unordered_map<std::string, std::unordered_map<Jetstream::Device, '
                f'std::vector<std::vector<U8>>>> {decorator}ShadersPackage = {{\n')

        for name in names:
            f.write('    { "' + name + '", { ')

            for target in targets:
                if target == 'msl':
                    device_str = 'Jetstream::Device::Metal'
                if target == 'spv':
                    device_str = 'Jetstream::Device::Vulkan'
                if target == 'wgsl':
                    device_str = 'Jetstream::Device::WebGPU'

                f.write('{ ' + device_str + ', { ')

                for type in types:
                    f.write(f'{name}_{target}_{type}_shader, ')

                f.write('}, }, ')

            f.write('}, },\n')

        f.write('};\n')

        # Package kernels.

        targets = set()
        names = set()

        for file in glob.glob(os.path.join(path, f'{stub}_*.comp.*')):
            target = file.split('.')[-1]
            targets.add(target)

            name = os.path.basename(file).split('.')[0].split('_')[1]
            names.add(name)

            varname = f'{name}_{target}_kernel'

            with open(os.path.join(path, f'{stub}_{name}.comp.{target}'), 'rb') as fr:
                data = fr.read()

            f.write('static const std::vector<U8> ' + varname + ' = {')
            f.write(','.join(f'0x{byte:02x}' for byte in data))
            f.write(',0x00};\n')

        f.write('\n')
        f.write(f'static std::unordered_map<std::string, std::unordered_map<Jetstream::Device, '
                f'std::vector<std::vector<U8>>>> {decorator}KernelsPackage = {{\n')

        for name in names:
            f.write('    { "' + name + '", { ')

            for target in targets:
                if target == 'msl':
                    device_str = 'Jetstream::Device::Metal'
                if target == 'spv':
                    device_str = 'Jetstream::Device::Vulkan'
                if target == 'wgsl':
                    device_str = 'Jetstream::Device::WebGPU'

                f.write('{ ' + device_str + ', { ')

                f.write(f'{name}_{target}_kernel, ')

                f.write('}, }, ')

            f.write('}, },\n')

        f.write('};\n')

if __name__ == '__main__':
    bin_to_header(sys.argv[1], sys.argv[2])
