#include "jetstream/benchmark.hh"
#include "jetstream/domains/visualization/signal_view/module.hh"
#include "jetstream/registry.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("signal_view") {
    return {
        {
            .variant = "lineplot-F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalView cfg;
                cfg.mode = "lineplot";
                return cfg;
            }())),
        },
        {
            .variant = "waterfall-F32-2048",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 2048),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalView cfg;
                cfg.mode = "waterfall";
                return cfg;
            }())),
        },
        {
            .variant = "combined-F32-2048",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", F32, 2048),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                SignalView cfg;
                cfg.mode = "lineplot_waterfall";
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
