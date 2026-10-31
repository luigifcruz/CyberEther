#include "jetstream/benchmark.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/complex_merge/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("complex_merge") {
    ComplexMerge rectangular;

    ComplexMerge polar;
    polar.mode = "polar";

    return {
        {
            .variant = "Rectangular-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("a", F32, 1024),
                JST_BENCHMARK_INPUT("b", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(rectangular),
        },
        {
            .variant = "Rectangular-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("a", F32, 8192),
                JST_BENCHMARK_INPUT("b", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(rectangular),
        },
        {
            .variant = "Rectangular-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("a", F32, 65536),
                JST_BENCHMARK_INPUT("b", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(rectangular),
        },
        {
            .variant = "Polar-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("a", F32, 1024),
                JST_BENCHMARK_INPUT("b", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(polar),
        },
        {
            .variant = "Polar-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("a", F32, 8192),
                JST_BENCHMARK_INPUT("b", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(polar),
        },
        {
            .variant = "Polar-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("a", F32, 65536),
                JST_BENCHMARK_INPUT("b", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(polar),
        },
    };
}

}  // namespace Jetstream::Modules
