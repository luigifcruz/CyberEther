#include <benchmark/benchmark.h>

#include "jstcore/fft/base.hpp"

using namespace Jetstream;

static void BM_SyncFFT(benchmark::State& state) {
  auto loop = Loop<Sync>::New();
  auto stream = std::vector<std::complex<float>>(2048);

  loop->add<FFT::CPU>("fft0", {}, {
    Data<VCF32>{Locale::CPU, stream},
  });

  for (auto _ : state) {
    loop->compute();
  }
}
BENCHMARK(BM_SyncFFT);

static void BM_AsyncFFT(benchmark::State& state) {
  auto loop = Loop<Async>::New();
  auto stream = std::vector<std::complex<float>>(2048);

  loop->add<FFT::CPU>("fft0", {}, {
    Data<VCF32>{Locale::CPU, stream},
  });

  for (auto _ : state) {
    loop->compute();
  }
}
BENCHMARK(BM_AsyncFFT);

BENCHMARK_MAIN();
