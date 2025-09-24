#include "jetstream/modules/multiply.hh"

namespace Jetstream {

template<template<Device, typename...> class Module, Device D, typename T>
void benchmark(ankerl::nanobench::Bench& bench, std::string name) {
    // 1D Alike
    JST_BENCHMARK_RUN("1024000 * 1024000 (1D-C/C) AL", {}, {
        .factorA = mem2::Tensor(D, mem2::TypeToDataType<T>(), {1024000}),
        .factorB = mem2::Tensor(D, mem2::TypeToDataType<T>(), {1024000}),
    }, T);

    // 2D (Contiguous/Contiguous) Alike
    JST_BENCHMARK_RUN("128x8000 * 128x8000 (2D-C/C) AL", {}, {
        .factorA = mem2::Tensor(D, mem2::TypeToDataType<T>(), {128, 8000}),
        .factorB = mem2::Tensor(D, mem2::TypeToDataType<T>(), {128, 8000}),
    }, T);

    // 2D (Non-Contiguous/Non-Contiguous) AL
    {
        mem2::Tensor factorA(D, mem2::TypeToDataType<T>(), {128, 1, 8000});
        mem2::Tensor factorB(D, mem2::TypeToDataType<T>(), {128, 1, 8000});

        factorA.slice({{}, 0, {}});
        factorB.slice({{}, 0, {}});

        JST_BENCHMARK_RUN("128x8000 * 128x8000 (2D-NC/NC) AL", {}, {
            .factorA = factorA,
            .factorB = factorB,
        }, T);
    }

    // Broadcast 2D (Contiguous/Non-Contiguous) 
    JST_BENCHMARK_RUN("128x8000 * 1x8000 (B-2D-C/NC)", {}, {
        .factorA = mem2::Tensor(D, mem2::TypeToDataType<T>(), {128, 8000}),
        .factorB = mem2::Tensor(D, mem2::TypeToDataType<T>(), {1, 8000}),
    }, T);

    // Broadcast 2D (Non-Contiguous/Non-Contiguous)
    {
        mem2::Tensor factorA(D, mem2::TypeToDataType<T>(), {128, 1, 8000});
        mem2::Tensor factorB(D, mem2::TypeToDataType<T>(), {1, 8000});

        factorA.slice({{}, 0, {}});

        JST_BENCHMARK_RUN("128x8000 * 1x8000 (B-2D-NC/NC)", {}, {
            .factorA = factorA,
            .factorB = factorB,
        }, T);
    }

    // 3D (Contiguous/Contiguous) Alike
    JST_BENCHMARK_RUN("2x64x8000 * 2x64x8000 (3D-C/C) AL", {}, {
        .factorA = mem2::Tensor(D, mem2::TypeToDataType<T>(), {2, 64, 8000}),
        .factorB = mem2::Tensor(D, mem2::TypeToDataType<T>(), {2, 64, 8000}),
    }, T);

    // 3D (Non-Contiguous/Non-Contiguous) AL
    {
        mem2::Tensor factorA(D, mem2::TypeToDataType<T>(), {2, 1, 64, 8000});
        mem2::Tensor factorB(D, mem2::TypeToDataType<T>(), {2, 1, 64, 8000});

        factorA.slice({{}, 0, {}, {}});
        factorB.slice({{}, 0, {}, {}});

        JST_BENCHMARK_RUN("2x64x8000 * 2x64x8000 (3D-NC/NC) AL", {}, {
            .factorA = factorA,
            .factorB = factorB,
        }, T);
    }

    // Broadcast 3D (Contiguous/Non-Contiguous)
    JST_BENCHMARK_RUN("2x64x8000 * 2x1x8000 (B-3D-C/NC)", {}, {
        .factorA = mem2::Tensor(D, mem2::TypeToDataType<T>(), {2, 64, 8000}),
        .factorB = mem2::Tensor(D, mem2::TypeToDataType<T>(), {2, 1, 8000}),
    }, T);
}

}  // namespace Jetstream