#include "jetstream/benchmark.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/complex_split/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("complex_split") {
    ComplexSplit rectangular;

    ComplexSplit polar;
    polar.mode = "polar";

    return {
        {
            .variant = "Rectangular-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(rectangular),
        },
        {
            .variant = "Rectangular-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(rectangular),
        },
        {
            .variant = "Rectangular-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(rectangular),
        },
        {
            .variant = "Polar-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(polar),
        },
        {
            .variant = "Polar-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(polar),
        },
        {
            .variant = "Polar-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(polar),
        },
    };
}

}  // namespace Jetstream::Modules
