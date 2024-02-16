project(
    'CyberEther',
    ['cpp'],
    version: '1.0.0-alpha4',
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
    ]
)

#
# Global Variables
#

jst_build_optimize = get_option('optimization')
jst_build_debug = get_option('debug')
jst_is_static = get_option('default_library') == 'static'
jst_is_release = not jst_build_debug
jst_is_crossbuild = meson.is_cross_build()

jst_is_linux   = target_machine.system() == 'linux' or \
                 target_machine.system() == 'freebsd'
jst_is_macos   = target_machine.system() == 'darwin'
jst_is_ios     = target_machine.system() == 'ios'
jst_is_android = target_machine.system() == 'android'
jst_is_browser = target_machine.system() == 'emscripten'
jst_is_windows = target_machine.system() == 'windows'

if not jst_is_linux   and \
   not jst_is_macos   and \
   not jst_is_ios     and \
   not jst_is_android and \
   not jst_is_browser and \
   not jst_is_windows
    error('Cannot determine target system. If your system is supported, file a bug report.')
endif

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

cfg_lst.set('version', meson.project_version())
cfg_lst.set('buildtype', get_option('buildtype'))
cfg_lst.set('optimization', get_option('optimization'))
cfg_lst.set('debug', get_option('debug'))
cfg_lst.set('native', get_option('native'))

cfg_lst.set('JST_IS_STATIC', jst_is_static)
cfg_lst.set('JST_IS_SHARED', not jst_is_static)

cfg_lst.set('JST_DEBUG_MODE', not jst_is_release)
cfg_lst.set('JST_RELEASE_MODE', jst_is_release)

cfg_lst.set('JST_OS_LINUX', jst_is_linux)
cfg_lst.set('JST_OS_MAC', jst_is_macos)
cfg_lst.set('JST_OS_IOS', jst_is_ios)
cfg_lst.set('JST_OS_ANDROID', jst_is_android)
cfg_lst.set('JST_OS_BROWSER', jst_is_browser)
cfg_lst.set('JST_OS_WINDOWS', jst_is_windows)

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
   not jst_is_ios
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
    'crossbuild': jst_is_crossbuild,
}, section: 'General', bool_yn: true)