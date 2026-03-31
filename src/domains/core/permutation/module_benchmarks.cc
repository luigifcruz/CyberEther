#include "jetstream/benchmark.hh"
#include "jetstream/domains/core/permutation/module.hh"

namespace Jetstream::Modules {

JST_BENCHMARKS("permutation") {
    Permutation transpose2d;
    transpose2d.permutation = {1, 0};

    Permutation reorder3d;
    reorder3d.permutation = {2, 0, 1};

    return {
        {
            .variant = "F32-2d-transpose",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 64, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(transpose2d),
        },
        {
            .variant = "F32-3d-reorder",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", F32, 8, 64, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(reorder3d),
        },
        {
            .variant = "CF32-2d-transpose",
            .inputs = {
                JST_BENCHMARK_INPUT("buffer", CF32, 64, 1024),
            },
            .config = JST_BENCHMARK_CONFIG(transpose2d),
        },
    };
}

}  // namespace Jetstream::Modules
