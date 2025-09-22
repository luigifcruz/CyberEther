#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/cast/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("cast") {
    Cast config;

    return {
        {
            .variant = "CI8-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CI8, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CI8-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CI8, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CI8-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CI8, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CI16-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CI16, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CI16-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CI16, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CI16-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CI16, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CU8-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CU8, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CU8-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CU8, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CU8-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CU8, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CU16-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CU16, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CU16-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CU16, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
        {
            .variant = "CU16-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CU16, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(config),
        },
    };
}

}  // namespace Jetstream::Modules
