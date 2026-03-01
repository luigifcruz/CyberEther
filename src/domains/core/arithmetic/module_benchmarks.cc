#include "jetstream/benchmark.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("arithmetic") {
    return {
        {
            .variant = "F32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 1024),
            },
            .config = {},
        },
        {
            .variant = "F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 8192),
            },
            .config = {},
        },
        {
            .variant = "F32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 65536),
            },
            .config = {},
        },
        {
            .variant = "CF32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1024),
            },
            .config = {},
        },
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = {},
        },
        {
            .variant = "CF32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 65536),
            },
            .config = {},
        },
    };
}

}  // namespace Jetstream::Modules
