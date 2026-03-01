#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/fm/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("fm") {
    return {
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                FM cfg;
                cfg.sampleRate = 240e3f;
                return cfg;
            }())),
        },
        {
            .variant = "CF32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                FM cfg;
                cfg.sampleRate = 240e3f;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
