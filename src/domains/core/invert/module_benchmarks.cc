#include "jetstream/benchmark.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/invert/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("invert") {
    Invert config;

    return {
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
