#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/am/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("am") {
    return {
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                AM cfg;
                cfg.sampleRate = 240e3f;
                cfg.dcAlpha = 0.995f;
                return cfg;
            }())),
        },
        {
            .variant = "CF32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                AM cfg;
                cfg.sampleRate = 240e3f;
                cfg.dcAlpha = 0.995f;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
