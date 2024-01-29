#include "jetstream/modules/amplitude.hh"

namespace Jetstream {

template<template<Device, typename...> class Module, Device D, typename IT, typename OT>
void benchmark(ankerl::nanobench::Bench& bench, std::string name) {
    JST_BENCHMARK_RUN("128x8000", {}, {
        .buffer = Tensor<D COMMA IT>({128 COMMA 8000}) COMMA
    }, IT, OT);
}

}  // namespace Jetstream