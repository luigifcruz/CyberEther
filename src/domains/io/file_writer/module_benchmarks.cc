#include "jetstream/benchmark.hh"
#include "jetstream/domains/io/file_writer/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("file_writer") {
    auto makeConfig = []() {
        FileWriter cfg;
        cfg.filepath = "/dev/null";
        cfg.overwrite = true;
        cfg.recording = true;
        return cfg;
    };

    return {
        {
            .variant = "F32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(makeConfig()),
        },
        {
            .variant = "F32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(makeConfig()),
        },
        {
            .variant = "F32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(makeConfig()),
        },
        {
            .variant = "CF32-1024",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(makeConfig()),
        },
        {
            .variant = "CF32-8192",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 8192),
            },
            .config = JST_BENCHMARK_CONFIG(makeConfig()),
        },
        {
            .variant = "CF32-65536",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 65536),
            },
            .config = JST_BENCHMARK_CONFIG(makeConfig()),
        },
    };
}

}  // namespace Jetstream::Modules
