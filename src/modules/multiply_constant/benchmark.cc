#include "jetstream/modules/multiply_constant.hh"

namespace Jetstream {

template<template<Device, typename...> class Module, Device D, typename T>
void benchmark(ankerl::nanobench::Bench& bench, std::string name) {
    JST_BENCHMARK_RUN("128x8000", {
        .constant = T(2),
    }, {
        .factor = mem2::Tensor(D, mem2::TypeToDataType<T>(), {128, 8000}),
    }, T);
}

}  // namespace Jetstream