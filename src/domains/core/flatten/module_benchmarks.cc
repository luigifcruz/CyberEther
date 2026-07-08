#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/flatten/module.hh"
#include "jetstream/registry.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("flatten") {
    Flatten config;

    return {
        {
            .variant = "F32-flatten-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 32, 32),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "F32-flatten-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 8, 32, 32),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CF32-flatten-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 32, 32),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
    };
}

}  // namespace Jetstream::Modules
