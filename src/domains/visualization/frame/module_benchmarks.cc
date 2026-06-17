#include "jetstream/benchmark.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/visualization/frame/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("frame") {
    return {
        {
            .variant = "F32-256x256",
            .inputs = {
                JST_BENCHMARK_INPUT("frame", F32, 256, 256),
            },
            .config = JST_BENCHMARK_CONFIG(Frame()),
        },
        {
            .variant = "F32-256x256-lut",
            .inputs = {
                JST_BENCHMARK_INPUT("frame", F32, 256, 256),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Frame cfg;
                cfg.lut = true;
                return cfg;
            }())),
        },
        {
            .variant = "F32-720p-rgb",
            .inputs = {
                JST_BENCHMARK_INPUT("frame", F32, 720, 1280, 3),
            },
            .config = JST_BENCHMARK_CONFIG(Frame()),
        },
    };
}

}  // namespace Jetstream::Modules
