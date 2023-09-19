#!/bin/python3

import os
import glob
import sys

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

def yml_to_header(path):
    with open(os.path.join(path, 'flowgraphs.hh'), 'w') as f:
        f.write('#pragma once\n\n')

        f.write('#include <tuple>\n')
        f.write('#include <unordered_map>\n\n')

        f.write('#include "jetstream/store.hh"\n\n')

        f.write('using namespace Jetstream;\n\n')
        
        f.write('static FlowgraphListStore DefaultFlowgraphListStore = {\n')

        for file in glob.glob(os.path.join(path, '..', 'flowgraphs', '*.yml')):
            filename = os.path.basename(file).split('.')[0]
            
            with open(file, 'r') as yml_file:
                if YAML_AVAILABLE:
                    yml_content = yaml.safe_load(yml_file)
                    title = yml_content.get('title', filename)
                    description = yml_content.get('description', 'No description.')
                else:
                    title = filename
                    description = 'No description.'
                
                yml_file.seek(0)
                content = yml_file.read().replace('\n', '\\n').replace('"', '\\"')
                
                f.write(f'    {{"{filename}", {{"{title}", "{description}", "{content}"}} }},\n')

        f.write('};\n')

if __name__ == '__main__':
    yml_to_header(sys.argv[1])
