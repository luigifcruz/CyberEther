#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/reshape/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("reshape") {
    Reshape config1;
    config1.shape = "[1024]";

    Reshape config2;
    config2.shape = "[128, 64]";

    Reshape config3;
    config3.shape = "[16, 16, 32]";

    Reshape config4;
    config4.shape = "[65536]";

    return {
        {
            .variant = "F32-flatten-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 32, 32),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "F32-unflatten-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config2),
        },
        {
            .variant = "F32-reshape-3d-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 128, 64),
            },
            .config = JST_BENCHMARK_CONFIG(config3),
        },
        {
            .variant = "F32-flatten-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 256, 256),
            },
            .config = JST_BENCHMARK_CONFIG(config4),
        },
        {
            .variant = "CF32-flatten-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 32, 32),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "CF32-unflatten-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config2),
        },
        {
            .variant = "CF32-flatten-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 256, 256),
            },
            .config = JST_BENCHMARK_CONFIG(config4),
        },
    };
}

}  // namespace Jetstream::Modules
