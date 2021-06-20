project(
    'glfw-meson-shim', 'cpp',
)

cmake   = import('cmake')
opt_var = cmake.subproject_options()

opt_var.add_cmake_defines({'BUILD_SHARED_LIBS': true})

opt_var.set_override_option('cpp_std', 'c++14')

sub_pro = cmake.subproject('glfw', options: opt_var)
glfw_dep = sub_pro.dependency('glfw')
