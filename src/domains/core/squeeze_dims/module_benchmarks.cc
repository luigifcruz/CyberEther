#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/squeeze_dims/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("squeeze_dims") {
    SqueezeDims config0;
    config0.axis = 0;

    SqueezeDims config1;
    config1.axis = 1;

    return {
        {
            .variant = "F32-2d-axis0-1x1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 1, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config0),
        },
        {
            .variant = "F32-2d-axis1-1024x1",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 1024, 1),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "F32-3d-axis0-1x128x64",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 1, 128, 64),
            },
            .config = JST_BENCHMARK_CONFIG(config0),
        },
        {
            .variant = "F32-3d-axis1-128x1x64",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 128, 1, 64),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "CF32-2d-axis0-1x1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config0),
        },
        {
            .variant = "CF32-3d-axis0-1x256x256",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1, 256, 256),
            },
            .config = JST_BENCHMARK_CONFIG(config0),
        },
    };
}

}  // namespace Jetstream::Modules
