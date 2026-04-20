test('parser', executable(
    'jetstream-parser-tests',
    [
        'main.cc',
        'conversions.cc',
        'serdes.cc',
        'hash.cc',
        'yaml.cc',
    ],
    dependencies: [libjetstream_dep, catch2_dep],
), is_parallel: false, timeout: 0)
