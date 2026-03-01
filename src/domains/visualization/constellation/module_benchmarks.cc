#include "jetstream/benchmark.hh"
#include "jetstream/domains/visualization/constellation/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("constellation") {
    return {
        {
            .variant = "CF32-2048",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 2048),
            },
            .config = {},
        },
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 8192),
            },
            .config = {},
        },
        {
            .variant = "CF32-128x2048",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 128, 2048),
            },
            .config = {},
        },
    };
}

}  // namespace Jetstream::Modules
