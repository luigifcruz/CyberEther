cmake = import('cmake')

opts = cmake.subproject_options()
opts.add_cmake_defines({
    'CMAKE_POSITION_INDEPENDENT_CODE': true,
    'RYML_BUILD_API': false,
    'RYML_BUILD_TESTS': false,
    'RYML_BUILD_BENCHMARKS': false,
    'RYML_BUILD_TOOLS': false,
    'RYML_INSTALL': false,
})

rapidyaml_proj = cmake.subproject('rapidyaml', options: opts, required: true)

if rapidyaml_proj.found()
    cfg_lst.set('JETSTREAM_LOADER_RAPIDYAML_AVAILABLE', true)
    dep_lst += rapidyaml_proj.dependency('ryml')
endif

ldr_lst += {'RapidYAML': rapidyaml_proj.found()}
