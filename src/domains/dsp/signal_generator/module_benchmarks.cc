#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/signal_generator/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("signal_generator") {
    return {
        {
            .variant = "sine-F32-1024",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalGenerator cfg;
                cfg.signalType = "sine";
                cfg.signalDataType = "F32";
                cfg.bufferSize = 1024;
                return cfg;
            }())),
        },
        {
            .variant = "sine-F32-8192",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalGenerator cfg;
                cfg.signalType = "sine";
                cfg.signalDataType = "F32";
                cfg.bufferSize = 8192;
                return cfg;
            }())),
        },
        {
            .variant = "sine-F32-65536",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalGenerator cfg;
                cfg.signalType = "sine";
                cfg.signalDataType = "F32";
                cfg.bufferSize = 65536;
                return cfg;
            }())),
        },
        {
            .variant = "sine-CF32-1024",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalGenerator cfg;
                cfg.signalType = "sine";
                cfg.signalDataType = "CF32";
                cfg.bufferSize = 1024;
                return cfg;
            }())),
        },
        {
            .variant = "sine-CF32-8192",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalGenerator cfg;
                cfg.signalType = "sine";
                cfg.signalDataType = "CF32";
                cfg.bufferSize = 8192;
                return cfg;
            }())),
        },
        {
            .variant = "sine-CF32-65536",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalGenerator cfg;
                cfg.signalType = "sine";
                cfg.signalDataType = "CF32";
                cfg.bufferSize = 65536;
                return cfg;
            }())),
        },
        {
            .variant = "noise-F32-8192",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalGenerator cfg;
                cfg.signalType = "noise";
                cfg.signalDataType = "F32";
                cfg.bufferSize = 8192;
                return cfg;
            }())),
        },
        {
            .variant = "noise-CF32-8192",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalGenerator cfg;
                cfg.signalType = "noise";
                cfg.signalDataType = "CF32";
                cfg.bufferSize = 8192;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
