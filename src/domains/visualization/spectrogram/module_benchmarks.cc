#include "jetstream/benchmark.hh"
#include "jetstream/domains/visualization/spectrogram/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("spectrogram") {
    return {
        {
            .variant = "F32-2048",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 2048),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Spectrogram cfg;
                cfg.height = 256;
                return cfg;
            }())),
        },
        {
            .variant = "F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Spectrogram cfg;
                cfg.height = 256;
                return cfg;
            }())),
        },
        {
            .variant = "F32-8192-h512",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Spectrogram cfg;
                cfg.height = 512;
                return cfg;
            }())),
        },
        {
            .variant = "F32-128x2048",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 128, 2048),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Spectrogram cfg;
                cfg.height = 256;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
