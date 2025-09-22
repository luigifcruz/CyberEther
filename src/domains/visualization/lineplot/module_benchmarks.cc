#include "jetstream/benchmark.hh"
#include "jetstream/domains/visualization/lineplot/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("lineplot") {
    return {
        {
            .variant = "F32-2048",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 2048),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Lineplot cfg;
                cfg.averaging = 1;
                cfg.decimation = 1;
                return cfg;
            }())),
        },
        {
            .variant = "F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Lineplot cfg;
                cfg.averaging = 1;
                cfg.decimation = 1;
                return cfg;
            }())),
        },
        {
            .variant = "F32-8192-avg16",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Lineplot cfg;
                cfg.averaging = 16;
                cfg.decimation = 1;
                return cfg;
            }())),
        },
        {
            .variant = "F32-128x2048",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 128, 2048),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Lineplot cfg;
                cfg.averaging = 1;
                cfg.decimation = 1;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
