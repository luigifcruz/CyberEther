#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/window/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("window") {
    return {
        {
            .variant = "1024",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                Window cfg;
                cfg.size = 1024;
                return cfg;
            }())),
        },
        {
            .variant = "65536",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                Window cfg;
                cfg.size = 65536;
                return cfg;
            }())),
        },
        {
            .variant = "8192",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                Window cfg;
                cfg.size = 8192;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
