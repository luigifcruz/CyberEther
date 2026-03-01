#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/overlap_add/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("overlap_add") {
    return {
        {
            .variant = "CF32-4x1024-ovl50",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 4, 1024),
                JST_BENCHMARK_INPUT("overlap", CF32, 4, 50),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                OverlapAdd cfg;
                cfg.axis = 1;
                return cfg;
            }())),
        },
        {
            .variant = "CF32-4x8192-ovl100",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 4, 8192),
                JST_BENCHMARK_INPUT("overlap", CF32, 4, 100),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                OverlapAdd cfg;
                cfg.axis = 1;
                return cfg;
            }())),
        },
        {
            .variant = "F32-4x1024-ovl50",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 4, 1024),
                JST_BENCHMARK_INPUT("overlap", F32, 4, 50),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                OverlapAdd cfg;
                cfg.axis = 1;
                return cfg;
            }())),
        },
        {
            .variant = "F32-4x8192-ovl100",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 4, 8192),
                JST_BENCHMARK_INPUT("overlap", F32, 4, 100),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                OverlapAdd cfg;
                cfg.axis = 1;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
