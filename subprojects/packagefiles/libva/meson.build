project(
    'libva',
    'c',
    version: '2.24.1',
    license: 'MIT',
    meson_version: '>= 0.60.0',
    default_options: [
        'c_std=gnu11',
        'warning_level=0',
    ],
)

cc = meson.get_compiler('c')

driverdir = ':'.join([
    '/usr/lib/x86_64-linux-gnu/dri',
    '/usr/lib/aarch64-linux-gnu/dri',
    '/usr/lib64/dri-nonfree',
    '/usr/lib64/dri-freeworld',
    '/usr/lib64/dri',
    '/usr/lib/dri',
    '/usr/local/lib/dri',
])

va_args = [
    '-DSYSCONFDIR="/etc"',
    '-DVA_DRIVERS_PATH="@0@"'.format(driverdir),
]
if cc.has_function('secure_getenv', prefix: '#define _GNU_SOURCE\n#include <stdlib.h>')
    va_args += ['-DHAVE_SECURE_GETENV']
endif

dl_dep = cc.find_library('dl', required: false)
libdrm_dep = dependency('libdrm', version: '>= 2.4.75')

subdir('va')
