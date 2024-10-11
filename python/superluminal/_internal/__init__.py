from superluminal._impl import *

#
# Constants
#

class constant:
    def __init__(self, key, value):
        self._key = key
        self._value = value

    def __repr__(self):
        return f'Constant(value={self._key})'

    @property
    def key(self):
        return self._key
    
    @property
    def value(self):
        return self._value
    
def _create_constants(names_list):
    for k, v in names_list.items():
        globals()[k] = constant(k, v)

_operations_lst = {
    'real': operation.real, 
    'imag': operation.imaginary,
    'amplitude': operation.amplitude,
    'phase': operation.phase,
}

_domains_lst = {
    'time': domain.time,
    'frequency': domain.frequency,
}

_types_lst = {
    'line': type.line,
    'heat': type.heat,
    'scatter': type.scatter,
    'waterfall': type.waterfall,
}

_create_constants(_operations_lst)
_create_constants(_domains_lst)
_create_constants(_types_lst)