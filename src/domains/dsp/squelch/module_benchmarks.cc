#include "jetstream/benchmark.hh"
#include "jetstream/registry.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("squelch") {
    return {
        {
            .variant = "F32-1024-closed",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 1024),
            },
            .config = {},
        },
        {
            .variant = "F32-8192-closed",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 8192),
            },
            .config = {},
        },
        {
            .variant = "F32-65536-closed",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 65536),
            },
            .config = {},
        },
        {
            .variant = "CF32-1024-closed",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 1024),
            },
            .config = {},
        },
        {
            .variant = "CF32-8192-closed",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 8192),
            },
            .config = {},
        },
        {
            .variant = "CF32-65536-closed",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 65536),
            },
            .config = {},
        },
    };
}

}  // namespace Jetstream::Modules
