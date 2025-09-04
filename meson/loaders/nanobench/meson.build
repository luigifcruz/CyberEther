cmake = import('cmake')

opts = cmake.subproject_options()
opts.add_cmake_defines({
    'NANOBENCH_BUILD_TESTS': false,
    'CMAKE_POSITION_INDEPENDENT_CODE': true,
})

nanobench_proj = cmake.subproject('nanobench', options: opts, required: true)

if nanobench_proj.found()
    cfg_lst.set('JETSTREAM_LOADER_NANOBENCH_AVAILABLE', true)
    dep_lst += nanobench_proj.dependency('nanobench')
endif

ldr_lst += {'Nanobench': nanobench_proj.found()}
