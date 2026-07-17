#include <blueprint/gain/module.hh>
#include <jetstream/benchmark.hh>
#include <jetstream/registry.hh>

namespace Jetstream::Modules {

JST_BENCHMARKS("blueprint_gain") {
    BlueprintGain config;
    config.gain = 2.0f;

    return {
        {
            .variant = "F32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "F32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CF32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CF32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
    };
}

}  // namespace Jetstream::Modules
