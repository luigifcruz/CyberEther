#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/filter_taps/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("filter_taps") {
    return {
        {
            .variant = "101-taps",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                FilterTaps cfg;
                cfg.sampleRate = 2.0e6;
                cfg.bandwidth = 1.0e6;
                cfg.center = {0.0};
                cfg.taps = 101;
                return cfg;
            }())),
        },
        {
            .variant = "501-taps",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                FilterTaps cfg;
                cfg.sampleRate = 2.0e6;
                cfg.bandwidth = 1.0e6;
                cfg.center = {0.0};
                cfg.taps = 501;
                return cfg;
            }())),
        },
        {
            .variant = "101-taps-offset",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                FilterTaps cfg;
                cfg.sampleRate = 2.0e6;
                cfg.bandwidth = 0.5e6;
                cfg.center = {0.25e6};
                cfg.taps = 101;
                return cfg;
            }())),
        },
        {
            .variant = "101-taps-3-heads",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                FilterTaps cfg;
                cfg.sampleRate = 2.0e6;
                cfg.bandwidth = 0.2e6;
                cfg.center = {0.0, 0.2e6, -0.4e6};
                cfg.taps = 101;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
