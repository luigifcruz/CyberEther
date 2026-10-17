sdr_hardware_test = executable(
    'jetstream-sdr-hardware-tests',
    'sdr.cc',
    dependencies: libjetstream_dep,
    build_by_default: false,
)

test(
    'sdr-hardware',
    sdr_hardware_test,
    suite: 'sdr',
    is_parallel: false,
    timeout: 1800,
)
