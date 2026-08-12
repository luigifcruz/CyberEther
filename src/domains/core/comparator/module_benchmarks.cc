#include "jetstream/benchmark.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/comparator/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("comparator") {
    Comparator config;
    config.inputCount = 2;

    return {
        {
            .variant = "F32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("input0", F32, 1024),
                JST_BENCHMARK_INPUT("input1", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("input0", F32, 8192),
                JST_BENCHMARK_INPUT("input1", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CF32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("input0", CF32, 1024),
                JST_BENCHMARK_INPUT("input1", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("input0", CF32, 8192),
                JST_BENCHMARK_INPUT("input1", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
    };
}

}  // namespace Jetstream::Modules
