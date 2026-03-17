project(
    'CyberEther',
    ['cpp'],
    version: '1.0.1',  # [VERSION HOOK]
    default_options: [
        'cpp_std=c++20',
        'buildtype=release',
        'b_ndebug=if-release',
        'warning_level=3',
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

jst_is_linux   = target_machine.system().startswith('linux') or \
                 target_machine.system().startswith('freebsd')
jst_is_macos   = target_machine.system().startswith('darwin')
jst_is_ios     = target_machine.system().startswith('ios')
jst_is_android = target_machine.system().startswith('android')
jst_is_browser = target_machine.system().startswith('emscripten')
jst_is_windows = target_machine.system().startswith('windows')

if not jst_is_linux   and \
   not jst_is_macos   and \
   not jst_is_ios     and \
   not jst_is_android and \
   not jst_is_browser and \
   not jst_is_windows
    error('Cannot determine target system. If your system is supported, file a bug report.')
endif

#
# Quirks
#

if jst_is_browser
    threads_dep = declare_dependency(compile_args: ['-pthread'], link_args: ['-pthread'])
    meson.override_dependency('threads', threads_dep)
endif

if get_option('native')
    add_project_arguments('-march=native', language: 'cpp')
endif

#
# Handle Options
#

cfg_lst = configuration_data()
src_lst = []
module_test_lst = []
dep_lst = [
    dependency('threads'),
]
whole_dep_lst = []
inc_lst = [
    include_directories('include'),
    include_directories('src'),
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
subdir('src')

#
# Library
#

if jst_is_browser
    libjetstream = static_library(
        'jetstream',
        src_lst,
        include_directories: inc_lst,
        dependencies: dep_lst + whole_dep_lst,
        install: true
    )

    libjetstream_dep = declare_dependency(
        include_directories: inc_lst,
        dependencies: dep_lst,
        link_whole: libjetstream
    )
else
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
endif

#
# Python Bindings
#

if get_option('python')
    subdir('python')
endif

#
# Install Library
#

subdir('include')

if not jst_is_static
    pkg = import('pkgconfig')
    pkg.generate(
        libjetstream,
        libraries: dep_lst,
    )
endif

#
# Executable
#

if jst_is_browser
    cyberether_exe = executable(
        'cyberether',
        'main.cc',
        include_directories: inc_lst,
        dependencies: libjetstream_dep,
        install: true,
    )
elif not jst_is_ios and \
     not jst_is_android
    cyberether_exe = executable(
        'cyberether',
        'main.cc',
        include_directories: inc_lst,
        dependencies: dep_lst,
        link_with: libjetstream,
        install: true,
    )
endif

#
# Optional
#

if get_option('tests') and not jst_is_browser
    subdir('tests')
endif

if get_option('examples') and not jst_is_browser
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
    'superluminal': cfg_lst.get('JETSTREAM_SUPERLUMINAL_AVAILABLE', false),
    'python bindings': cfg_lst.get('JETSTREAM_SUPERLUMINAL_PYTHON_AVAILABLE', false),
}, section: 'General', bool_yn: true)
