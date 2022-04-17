#include <benchmark/benchmark.h>

#include "jstcore/fft/base.hpp"

using namespace Jetstream;

static void BM_SyncFFT(benchmark::State& state) {
    auto stream = std::vector<std::complex<float>>(2048);

    Jetstream::Stream({
        Jetstream::Block<FFT::CPU>({}, {
            Data<VCF32>{Locale::CPU, stream},
        }),
    });

    for (auto _ : state) {
        Jetstream::Compute();
    }
}
BENCHMARK(BM_SyncFFT);

BENCHMARK_MAIN();
