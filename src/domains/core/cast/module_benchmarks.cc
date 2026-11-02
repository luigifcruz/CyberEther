#include "jetstream/benchmark.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/cast/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("cast") {
    Cast config;

    Cast toCi8;
    toCi8.outputType = "CI8";

    Cast toCi16;
    toCi16.outputType = "CI16";

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
        {
            .variant = "CF32-CI8-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(toCi8),
        },
        {
            .variant = "CF32-CI8-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(toCi8),
        },
        {
            .variant = "CF32-CI8-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(toCi8),
        },
        {
            .variant = "CF32-CI16-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(toCi16),
        },
        {
            .variant = "CF32-CI16-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(toCi16),
        },
        {
            .variant = "CF32-CI16-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(toCi16),
        },
    };
}

}  // namespace Jetstream::Modules
