#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/fft/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("fft") {
    return {
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Fft cfg;
                cfg.forward = true;
                return cfg;
            }())),
        },
        {
            .variant = "CF32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Fft cfg;
                cfg.forward = true;
                return cfg;
            }())),
        },
        {
            .variant = "F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Fft cfg;
                cfg.forward = true;
                return cfg;
            }())),
        },
        {
            .variant = "F32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Fft cfg;
                cfg.forward = true;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
