#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/pad/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("pad") {
    Pad config1;
    config1.size = 512;
    config1.axis = 0;

    Pad config2;
    config2.size = 4096;
    config2.axis = 0;

    Pad config3;
    config3.size = 32768;
    config3.axis = 0;

    return {
        {
            .variant = "F32-1024-pad512",
            .inputs = {
                JST_BENCHMARK_INPUT("unpadded", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "F32-8192-pad4096",
            .inputs = {
                JST_BENCHMARK_INPUT("unpadded", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config2),
        },
        {
            .variant = "F32-65536-pad32768",
            .inputs = {
                JST_BENCHMARK_INPUT("unpadded", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config3),
        },
        {
            .variant = "CF32-1024-pad512",
            .inputs = {
                JST_BENCHMARK_INPUT("unpadded", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "CF32-8192-pad4096",
            .inputs = {
                JST_BENCHMARK_INPUT("unpadded", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config2),
        },
        {
            .variant = "CF32-65536-pad32768",
            .inputs = {
                JST_BENCHMARK_INPUT("unpadded", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config3),
        },
    };
}

}  // namespace Jetstream::Modules
