#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/throttle/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("throttle") {
    return {
        {
            .variant = "F32-1024-10ms",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Throttle cfg;
                cfg.intervalMs = 10;
                return cfg;
            }())),
        },
        {
            .variant = "F32-8192-10ms",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Throttle cfg;
                cfg.intervalMs = 10;
                return cfg;
            }())),
        },
        {
            .variant = "CF32-8192-10ms",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Throttle cfg;
                cfg.intervalMs = 10;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
