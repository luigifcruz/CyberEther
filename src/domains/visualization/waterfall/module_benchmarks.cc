#include "jetstream/benchmark.hh"
#include "jetstream/domains/visualization/waterfall/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("waterfall") {
    return {
        {
            .variant = "F32-2048",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 2048),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Waterfall cfg;
                cfg.height = 512;
                cfg.interpolate = true;
                return cfg;
            }())),
        },
        {
            .variant = "F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Waterfall cfg;
                cfg.height = 512;
                cfg.interpolate = true;
                return cfg;
            }())),
        },
        {
            .variant = "F32-8192-h1024",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Waterfall cfg;
                cfg.height = 1024;
                cfg.interpolate = true;
                return cfg;
            }())),
        },
        {
            .variant = "F32-128x2048",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 128, 2048),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Waterfall cfg;
                cfg.height = 512;
                cfg.interpolate = true;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
