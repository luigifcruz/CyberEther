#include "jetstream/modules/duplicate.hh"

namespace Jetstream {

template<template<Device, typename...> class Module, Device D, typename IT>
void benchmark(ankerl::nanobench::Bench& bench, std::string name) {
    // Contiguous
    JST_BENCHMARK_RUN("128x8000 (C)", {}, {
        .buffer = mem2::Tensor(D, mem2::TypeToDataType<IT>(), {128, 8000}),
    }, IT);

    // Non-Contiguous
    {
        mem2::Tensor buffer(D, mem2::TypeToDataType<IT>(), {128, 16000});
        buffer.slice({{}, {0, 0, 2}});
        JST_BENCHMARK_RUN("128x8000 (NC)", {}, {
            .buffer = buffer,
        }, IT);
    }
}

}  // namespace Jetstream