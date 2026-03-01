#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/rrc_filter/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("rrc_filter") {
    return {
        {
            .variant = "CF32-8192-101taps",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                RrcFilter cfg;
                cfg.symbolRate = 1.0e6f;
                cfg.sampleRate = 4.0e6f;
                cfg.rollOff = 0.35f;
                cfg.taps = 101;
                return cfg;
            }())),
        },
        {
            .variant = "CF32-8192-11taps",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                RrcFilter cfg;
                cfg.symbolRate = 1.0e6f;
                cfg.sampleRate = 4.0e6f;
                cfg.rollOff = 0.35f;
                cfg.taps = 11;
                return cfg;
            }())),
        },
        {
            .variant = "CF32-65536-101taps",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                RrcFilter cfg;
                cfg.symbolRate = 1.0e6f;
                cfg.sampleRate = 4.0e6f;
                cfg.rollOff = 0.35f;
                cfg.taps = 101;
                return cfg;
            }())),
        },
        {
            .variant = "F32-8192-101taps",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                RrcFilter cfg;
                cfg.symbolRate = 1.0e6f;
                cfg.sampleRate = 4.0e6f;
                cfg.rollOff = 0.35f;
                cfg.taps = 101;
                return cfg;
            }())),
        },
        {
            .variant = "F32-65536-101taps",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                RrcFilter cfg;
                cfg.symbolRate = 1.0e6f;
                cfg.sampleRate = 4.0e6f;
                cfg.rollOff = 0.35f;
                cfg.taps = 101;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
