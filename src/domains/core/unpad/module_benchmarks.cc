#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/unpad/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("unpad") {
    Unpad config1;
    config1.size = 512;
    config1.axis = 0;

    Unpad config2;
    config2.size = 4096;
    config2.axis = 0;

    Unpad config3;
    config3.size = 32768;
    config3.axis = 0;

    return {
        {
            .variant = "F32-1536-unpad512",
            .inputs = {
                JST_BENCHMARK_INPUT("padded", F32, 1536),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "F32-12288-unpad4096",
            .inputs = {
                JST_BENCHMARK_INPUT("padded", F32, 12288),
            },
            .config = JST_BENCHMARK_CONFIG(config2),
        },
        {
            .variant = "F32-98304-unpad32768",
            .inputs = {
                JST_BENCHMARK_INPUT("padded", F32, 98304),
            },
            .config = JST_BENCHMARK_CONFIG(config3),
        },
        {
            .variant = "CF32-1536-unpad512",
            .inputs = {
                JST_BENCHMARK_INPUT("padded", CF32, 1536),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "CF32-12288-unpad4096",
            .inputs = {
                JST_BENCHMARK_INPUT("padded", CF32, 12288),
            },
            .config = JST_BENCHMARK_CONFIG(config2),
        },
        {
            .variant = "CF32-98304-unpad32768",
            .inputs = {
                JST_BENCHMARK_INPUT("padded", CF32, 98304),
            },
            .config = JST_BENCHMARK_CONFIG(config3),
        },
    };
}

}  // namespace Jetstream::Modules
