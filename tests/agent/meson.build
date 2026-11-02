agent_test_domains = {
    'tools': ['tools/flowgraph.cc'],
    'session': ['session/session.cc', 'session/harness.cc'],
    'provider': ['provider/provider.cc', 'provider/requests.cc', 'provider/replay.cc', 'provider/retry.cc',
                 'provider/live.cc'],
}

agent_test_targets = []
foreach domain, sources : agent_test_domains
    agent_tests = executable(
        'jetstream-agent-' + domain + '-tests',
        ['../support/catch_main.cc'] + sources,
        dependencies: [libjetstream_dep, catch2_dep],
        include_directories: include_directories('../support'),
    )
    agent_test_targets += [agent_tests]
    test('agent-' + domain, agent_tests, suite: 'agent', is_parallel: false, timeout: 60)
endforeach

alias_target('jetstream-agent-tests', agent_test_targets)
