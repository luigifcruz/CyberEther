#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/expand_dims/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("expand_dims") {
    ExpandDims config0;
    config0.axis = 0;

    ExpandDims config1;
    config1.axis = 1;

    return {
        {
            .variant = "F32-1d-axis0-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config0),
        },
        {
            .variant = "F32-1d-axis1-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "F32-2d-axis0-128x64",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 128, 64),
            },
            .config = JST_BENCHMARK_CONFIG(config0),
        },
        {
            .variant = "F32-2d-axis1-128x64",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 128, 64),
            },
            .config = JST_BENCHMARK_CONFIG(config1),
        },
        {
            .variant = "CF32-1d-axis0-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(config0),
        },
        {
            .variant = "CF32-2d-axis0-256x256",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 256, 256),
            },
            .config = JST_BENCHMARK_CONFIG(config0),
        },
    };
}

}  // namespace Jetstream::Modules
