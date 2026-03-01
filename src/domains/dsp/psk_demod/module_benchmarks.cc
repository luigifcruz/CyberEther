#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/psk_demod/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("psk_demod") {
    return {
        {
            .variant = "QPSK-CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                PskDemod cfg;
                cfg.pskType = "qpsk";
                cfg.sampleRate = 2000000.0;
                cfg.symbolRate = 500000.0;
                cfg.frequencyLoopBandwidth = 0.05;
                cfg.timingLoopBandwidth = 0.05;
                cfg.dampingFactor = 0.707;
                return cfg;
            }())),
        },
        {
            .variant = "QPSK-CF32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                PskDemod cfg;
                cfg.pskType = "qpsk";
                cfg.sampleRate = 2000000.0;
                cfg.symbolRate = 500000.0;
                cfg.frequencyLoopBandwidth = 0.05;
                cfg.timingLoopBandwidth = 0.05;
                cfg.dampingFactor = 0.707;
                return cfg;
            }())),
        },
        {
            .variant = "BPSK-CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                PskDemod cfg;
                cfg.pskType = "bpsk";
                cfg.sampleRate = 1000000.0;
                cfg.symbolRate = 250000.0;
                cfg.frequencyLoopBandwidth = 0.05;
                cfg.timingLoopBandwidth = 0.05;
                cfg.dampingFactor = 0.707;
                return cfg;
            }())),
        },
        {
            .variant = "8PSK-CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                PskDemod cfg;
                cfg.pskType = "8psk";
                cfg.sampleRate = 4000000.0;
                cfg.symbolRate = 1000000.0;
                cfg.frequencyLoopBandwidth = 0.05;
                cfg.timingLoopBandwidth = 0.05;
                cfg.dampingFactor = 0.707;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
