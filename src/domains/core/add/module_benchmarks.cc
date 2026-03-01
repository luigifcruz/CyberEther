#include "jetstream/benchmark.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("add") {
    return {
        {
            .variant = "F32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("a", F32, 1024),
                JST_BENCHMARK_INPUT("b", F32, 1024),
            },
            .config = {},
        },
        {
            .variant = "F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("a", F32, 8192),
                JST_BENCHMARK_INPUT("b", F32, 8192),
            },
            .config = {},
        },
        {
            .variant = "F32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("a", F32, 65536),
                JST_BENCHMARK_INPUT("b", F32, 65536),
            },
            .config = {},
        },
        {
            .variant = "CF32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("a", CF32, 1024),
                JST_BENCHMARK_INPUT("b", CF32, 1024),
            },
            .config = {},
        },
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("a", CF32, 8192),
                JST_BENCHMARK_INPUT("b", CF32, 8192),
            },
            .config = {},
        },
        {
            .variant = "CF32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("a", CF32, 65536),
                JST_BENCHMARK_INPUT("b", CF32, 65536),
            },
            .config = {},
        },
    };
}

}  // namespace Jetstream::Modules
