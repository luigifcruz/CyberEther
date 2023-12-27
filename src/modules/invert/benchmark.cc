#include "jetstream/modules/invert.hh"

namespace Jetstream {

template<template<Device, typename...> class Module, Device D, typename T>
void benchmark(ankerl::nanobench::Bench& bench, std::string name) {
    JST_BENCHMARK_RUN("128x8000", {}, {
        .buffer = Tensor<D COMMA T>({128 COMMA 8000}) COMMA
    }, T);
}

}  // namespace Jetstream