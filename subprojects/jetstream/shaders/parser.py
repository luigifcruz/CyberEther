#!/bin/python3

import os
import glob
import sys

def bin_to_header(path, stub):
    path = os.path.join(path, 'shaders')
    
    with open(os.path.join(path, f'{stub}_shaders.hh'), 'w') as f:
        f.write('#pragma once\n')
        f.write('#include <array>\n')

        for file in glob.glob(os.path.join(path, f'{stub}_*.vert.*')):
            target = file.split('.')[-1]
            name = os.path.basename(file).split('.')[0].split('_')[1]

            for type in ['frag', 'vert']:
                varname = f'{name}_{target}_{type}_shader'

                with open(os.path.join(path, f'{stub}_{name}.{type}.{target}'), 'rb') as fr:
                    data = fr.read()
                size = str(len(data) + 1)

                f.write('static const std::array<unsigned char, ' + size + '> ' + varname + ' = {')
                f.write(','.join(f'0x{byte:02x}' for byte in data))
                f.write(',0x00};\n')

if __name__ == '__main__':
    bin_to_header(sys.argv[1], sys.argv[2])