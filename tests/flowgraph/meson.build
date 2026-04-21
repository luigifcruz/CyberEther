test('flowgraph', executable(
    'jetstream-flowgraph-tests',
    [
        'main.cc',
        'api.cc',
        'flowgraph.cc',
    ],
    dependencies: [libjetstream_dep, catch2_dep],
    include_directories: include_directories('../common'),
), is_parallel: false, timeout: 0)
