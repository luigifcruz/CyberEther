sdr_hardware_test = executable(
    'jetstream-sdr-hardware-tests',
    'sdr.cc',
    dependencies: libjetstream_dep,
    link_args: test_link_args,
    link_depends: browser_exe_link_depends,
    build_by_default: false,
)

if jst_is_browser
    test('sdr-hardware-wasm', node,
         args: [files('../browser/run.cjs'), sdr_hardware_test.full_path(), '--usb', '--timeout', '1740000'],
         depends: sdr_hardware_test,
         suite: ['sdr', 'wasm'], is_parallel: false, timeout: 1800)
else
    test('sdr-hardware', sdr_hardware_test,
         suite: 'sdr', is_parallel: false, timeout: 1800)
endif
