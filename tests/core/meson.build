core_test_dependencies = [libjetstream_dep, catch2_dep]
core_test_includes = include_directories('../support')
core_test_main = '../support/catch_main.cc'

core_test_domains = {
    'application': ['application/instance_backend.cc'],
    'compositor': ['compositor/compositor.cc'],
    'extensions': [
        'extensions/plugins.cc',
        'extensions/benchmark.cc',
    ],
    'flowgraph': [
        'flowgraph/api.cc',
        'flowgraph/execution.cc',
        'flowgraph/mutation.cc',
        'flowgraph/state.cc',
    ],
    'lifecycle': ['lifecycle/block_module.cc'],
    'memory': [
        'memory/axis.cc',
        'memory/token.cc',
        'memory/snapshot.cc',
        'memory/buffer.cc',
        'memory/tensor.cc',
    ],
    'platform': [
        'platform/paths.cc',
        'platform/storage.cc',
    ],
    'registry': ['registry/registry.cc'],
    'render': ['render/transfers.cc'],
    'runtime': [
        'runtime/scheduler.cc',
        'runtime/native.cc',
        'runtime/python.cc',
    ],
    'serialization': [
        'serialization/conversions.cc',
        'serialization/serdes.cc',
        'serialization/hash.cc',
        'serialization/yaml.cc',
    ],
    'settings': ['settings/settings.cc'],
    'surface': ['surface/surface.cc'],
    'tools': ['tools/circular_buffer.cc'],
}

core_test_targets = []
foreach domain, sources : core_test_domains
    domain_tests = executable(
        'jetstream-core-' + domain + '-tests',
        [core_test_main] + sources,
        dependencies: core_test_dependencies,
        include_directories: core_test_includes,
    )
    core_test_targets += [domain_tests]
    test('core-' + domain, domain_tests,
         suite: 'core', is_parallel: false, timeout: 180)
endforeach

# TODO: Re-enable after CLI output capture supports separate Windows static CRTs.
if not jst_is_ios and not jst_is_android and not jst_is_windows
    core_integration_tests = executable(
        'jetstream-core-integration-tests',
        'integration/cli.cc',
        dependencies: core_test_dependencies,
        include_directories: core_test_includes,
    )
    core_test_targets += [core_integration_tests]
    test('core-integration', core_integration_tests,
         suite: 'core', is_parallel: false, timeout: 60)
endif

alias_target('jetstream-core-tests', core_test_targets)
