project(
    'CyberEther',
    ['cpp'],
    version: '1.0.0',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
    ]
)

#
# Global Variables
#

jst_is_static = get_option('default_library') == 'static'
jst_is_release = get_option('buildtype') == 'release'

jst_is_linux   = target_machine.system() == 'linux'
jst_is_macos   = target_machine.system() == 'darwin'
jst_is_ios     = target_machine.system() == 'ios'
jst_is_android = target_machine.system() == 'android'
jst_is_browser = target_machine.system() == 'emscripten'
jst_is_windows = target_machine.system() == 'windows'

#
# Handle Options
#

if get_option('native')
    add_project_arguments('-march=native', language: 'cpp')
endif

cfg_lst = configuration_data()
src_lst = []
dep_lst = [
    dependency('threads'),
    dependency('fmt', version: '>=9.1', fallback: 'fmt', default_options: ['default_library=static']),
]
inc_lst = [
    include_directories('include'),
    include_directories('.'),
    include_directories('include/jetstream/backend/devices/metal'),
    include_directories('include/jetstream/render/tools'),
]

#
# Handle Configuration Data
#

cfg_lst.set('buildtype', get_option('buildtype'))

#
# Call Subdirectories
#

subdir('meson')
subdir('resources')
subdir('flowgraphs')
subdir('shaders')
subdir('src')
subdir('include')

#
# Declare Library
#

libjetstream = library(
    'jetstream',
    src_lst,
    include_directories: inc_lst,
    dependencies: dep_lst,
    install: true
)

libjetstream_dep = declare_dependency(
    include_directories: inc_lst,
    dependencies: dep_lst,
    link_with: libjetstream
)

if not jst_is_static
    pkg = import('pkgconfig')
    pkg.generate(
        libjetstream,
        libraries: dep_lst,
    )
endif

if not jst_is_ios and \
   not jst_is_android
    executable(
        'cyberether', 
        'main.cc',
        dependencies: libjetstream_dep,
        install: true,
    )
endif

#
# Optionals
#

if not jst_is_browser and \
   not jst_is_android and \
   not jst_is_ios and \
   not jst_is_windows
    subdir('tests')
endif


if not jst_is_browser and \
   not jst_is_android and \
   not jst_is_ios
    subdir('examples')
endif

#
# Print Summary
#

summary({
    'cpp_std': get_option('cpp_std'),
    'prefix': get_option('prefix'),
    'buildtype': get_option('buildtype'),
    'crossbuild': meson.is_cross_build(),
}, section: 'General', bool_yn: true)