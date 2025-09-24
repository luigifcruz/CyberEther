#include "jetstream/modules/amplitude.hh"

namespace Jetstream {

template<template<Device, typename...> class Module, Device D, typename IT, typename OT>
void benchmark(ankerl::nanobench::Bench& bench, std::string name) {
    JST_BENCHMARK_RUN("128x8000", {}, {
        .buffer = mem2::Tensor(D, mem2::TypeToDataType<IT>(), {128, 8000}),
    }, IT, OT);
}

}  // namespace Jetstream