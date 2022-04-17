#include <benchmark/benchmark.h>
#include "jetstream/base.hpp"

using namespace Jetstream;

class Dummy : public Module {
public:
  class Config {};
  class Input {};

  Dummy(const Config& config, const Input& input) {}
};

static void BM_SyncLoop(benchmark::State& state) {
    Jetstream::Stream({
        Jetstream::Block<Dummy>({}, {}),
    });

    for (auto _ : state) {
        Jetstream::Compute();
    }
}
BENCHMARK(BM_SyncLoop);

static void BM_ComputePresentLoop(benchmark::State& state) {
    Jetstream::Stream({
        Jetstream::Block<Dummy>({}, {}),
    });

    std::size_t ITER = 8;

    for (auto _ : state) {
        auto thread = std::thread([&]{
           for (int i = 0; i < ITER; i++) {
                Jetstream::Compute();
            } 
        });

        for (int i = 0; i < ITER; i++) {
            Jetstream::Present();
        }

        thread.join();
    }

}
BENCHMARK(BM_ComputePresentLoop);

BENCHMARK_MAIN();
