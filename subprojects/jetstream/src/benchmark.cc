#include <benchmark/benchmark.h>
#include "jetstream/base.hpp"

using namespace Jetstream;

class Dummy : public Module {
public:
  class Config {};
  class Input {};

  Dummy(const Config & config, const Input & input) {}
};

static void BM_SyncLoop(benchmark::State& state) {
  auto loop = Loop<Sync>::New();
  loop->add<Dummy>("dummy0", {}, {});
  for (auto _ : state) {
    loop->compute();
  }
}
BENCHMARK(BM_SyncLoop);

static void BM_AsyncLoop(benchmark::State& state) {
  auto loop = Loop<Async>::New();
  loop->add<Dummy>("dummy0", {}, {});
  for (auto _ : state) {
    loop->compute();
  }
}
BENCHMARK(BM_AsyncLoop);

static void BM_SyncSyncLoop(benchmark::State& state) {
  auto loop = Loop<Sync>::New();
  loop->add<Dummy>("dummy0", {}, {});
  auto sync = Subloop<Sync>::New(loop);
  sync->add<Dummy>("dummy1", {}, {});
  for (auto _ : state) {
    loop->compute();
  }
}
BENCHMARK(BM_SyncSyncLoop);

static void BM_SyncAsyncLoop(benchmark::State& state) {
  auto loop = Loop<Sync>::New();
  loop->add<Dummy>("dummy0", {}, {});
  auto async = Subloop<Async>::New(loop);
  async->add<Dummy>("dummy1", {}, {});
  for (auto _ : state) {
    loop->compute();
  }
}
BENCHMARK(BM_SyncAsyncLoop);

BENCHMARK_MAIN();
