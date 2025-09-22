#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/slice/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("slice") {
    Slice config1;
    config1.slice = "[0:512]";

    Slice config2;
    config2.slice = "[0:4096]";

    Slice config3;
    config3.slice = "[0:32768]";

    Slice config4;
    config4.slice = "[::2]";

    return {
        {
            .variant = "F32-1024-range",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "F32-8192-range",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config2),
        },
        {
            .variant = "F32-65536-range",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config3),
        },
        {
            .variant = "F32-65536-step",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config4),
        },
        {
            .variant = "CF32-1024-range",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "CF32-8192-range",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config2),
        },
        {
            .variant = "CF32-65536-range",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config3),
        },
    };
}

}  // namespace Jetstream::Modules
