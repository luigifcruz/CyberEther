#!/bin/python3

import os
import sys

def create_header(path, output, inputs):
    with open(os.path.join(path, output), 'w') as fh:
        fh.write('#pragma once\n\n')

        fh.write('#include <stdint.h>\n\n')

        fh.write('namespace Jetstream::Resources {\n\n')

        for file in inputs:
            filename = os.path.basename(file).split('.')[0]
            
            with open(os.path.join(path, file), 'rb') as f:
                title = filename
                content = f.read()
                size = len(content)

                hex_data = ["0x{:02x}".format(byte) for byte in content]
                formatted_hex_data = ", ".join(hex_data)
                
                fh.write(f'static const uint8_t {title}_bin[] = {{{formatted_hex_data}}};\n')
                fh.write(f'static const uint32_t {title}_len = {size};\n\n')

        fh.write('\n}  // namespace Jetstream::Resources\n')

if __name__ == '__main__':
    create_header(sys.argv[1], sys.argv[2], sys.argv[3:])
