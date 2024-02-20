#include "jetstream/modules/duplicate.hh"

namespace Jetstream {

template<template<Device, typename...> class Module, Device D, typename IT>
void benchmark(ankerl::nanobench::Bench& bench, std::string name) {
    // Contiguous
    JST_BENCHMARK_RUN("128x8000 (C)", {}, {
        .buffer = Tensor<D COMMA IT>({128 COMMA 8000}) COMMA
    }, IT);

    // Non-Contiguous
    {
        Tensor<D, IT> buffer({128, 16000});
        buffer.slice({{}, {0, 0, 2}});
        JST_BENCHMARK_RUN("128x8000 (NC)", {}, {
            .buffer = buffer COMMA
        }, IT);
    }
}

}  // namespace Jetstream