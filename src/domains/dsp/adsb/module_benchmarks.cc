#include "jetstream/benchmark.hh"
#include "jetstream/domains/dsp/adsb/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("adsb") {
    return {
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Adsb cfg;
                return cfg;
            }())),
        },
        {
            .variant = "CF32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("signal", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(([]{
                Adsb cfg;
                return cfg;
            }())),
        },
    };
}

}  // namespace Jetstream::Modules
