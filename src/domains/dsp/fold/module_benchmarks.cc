#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/fold/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("fold") {
    return {
        {
            .variant = "CF32-8192-to-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Fold cfg;
                cfg.axis = 0;
                cfg.offset = 0;
                cfg.size = 1024;
                return cfg;
            }())),
        },
        {
            .variant = "CF32-65536-to-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Fold cfg;
                cfg.axis = 0;
                cfg.offset = 0;
                cfg.size = 1024;
                return cfg;
            }())),
        },
        {
            .variant = "F32-8192-to-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Fold cfg;
                cfg.axis = 0;
                cfg.offset = 0;
                cfg.size = 1024;
                return cfg;
            }())),
        },
        {
            .variant = "F32-65536-to-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Fold cfg;
                cfg.axis = 0;
                cfg.offset = 0;
                cfg.size = 1024;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
