#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/ones_tensor/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("ones_tensor") {
    return {
        {
            .variant = "f32_8192",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                OnesTensor cfg;
                cfg.shape = {8192};
                cfg.dataType = "F32";
                return cfg;
            }())),
        },
        {
            .variant = "cf32_256x256",
            .inputs = {},
            .config = JST_BENCHMARK_CONFIG(([]{
                OnesTensor cfg;
                cfg.shape = {256, 256};
                cfg.dataType = "CF32";
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
