#!/bin/python3

import os
import glob
import sys

def bin_to_header(path, stub):
    path = os.path.join(path, 'shaders')
    
    shader_target = []
    for file in glob.glob(os.path.join(path, f'{stub}.vert.*')):
        shader_target.append(file.split('.')[-1])

    with open(os.path.join(path, f'{stub}_shaders.hh'), 'w') as f:
        f.write('#pragma once\n')
        f.write('#include <array>\n')

        for target in shader_target:
            for type in ['frag', 'vert']:
                varname = f'{stub}_{target}_{type}_shader'

                with open(os.path.join(path, f'{stub}.{type}.{target}'), 'rb') as fr:
                    data = fr.read()
                size = str(len(data))

                f.write('static const std::array<unsigned char, ' + size + '> ' + varname + ' = {')
                f.write(','.join(f'0x{byte:02x}' for byte in data))
                f.write('};\n')

if __name__ == '__main__':
    bin_to_header(sys.argv[1], sys.argv[2])