#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/multiply_constant/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("multiply_constant") {
    MultiplyConstant config;
    config.constant = 2.0f;

    return {
        {
            .variant = "F32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("factor", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("factor", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "F32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("factor", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CF32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("factor", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("factor", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CF32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("factor", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
    };
}

}  // namespace Jetstream::Modules
