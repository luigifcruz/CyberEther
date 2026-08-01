#include "jetstream/benchmark.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/overlap_add/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("overlap_add") {
    return {
        {
            .variant = "CF32-1024-ovl50",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1024),
                JST_BENCHMARK_INPUT("overlap", CF32, 50),
            },
            .config = JST_BENCHMARK_CONFIG(OverlapAdd{}),
        },
        {
            .variant = "CF32-8192-ovl100",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
                JST_BENCHMARK_INPUT("overlap", CF32, 100),
            },
            .config = JST_BENCHMARK_CONFIG(OverlapAdd{}),
        },
        {
            .variant = "F32-1024-ovl50",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 1024),
                JST_BENCHMARK_INPUT("overlap", F32, 50),
            },
            .config = JST_BENCHMARK_CONFIG(OverlapAdd{}),
        },
        {
            .variant = "F32-8192-ovl100",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 8192),
                JST_BENCHMARK_INPUT("overlap", F32, 100),
            },
            .config = JST_BENCHMARK_CONFIG(OverlapAdd{}),
        },
    };
}

}  // namespace Jetstream::Modules
