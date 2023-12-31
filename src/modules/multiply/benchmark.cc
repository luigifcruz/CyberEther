#include "jetstream/modules/multiply.hh"

namespace Jetstream {

template<template<Device, typename...> class Module, Device D, typename T>
void benchmark(ankerl::nanobench::Bench& bench, std::string name) {
    // 1D Alike
    JST_BENCHMARK_RUN("1024000 * 1024000 (1D-C/C) AL", {}, {
        .factorA = Tensor<D COMMA T>({1024000}) COMMA
        .factorB = Tensor<D COMMA T>({1024000}) COMMA
    }, T);

    // 2D (Contiguous/Contiguous) Alike
    JST_BENCHMARK_RUN("128x8000 * 128x8000 (2D-C/C) AL", {}, {
        .factorA = Tensor<D COMMA T>({128 COMMA 8000}) COMMA
        .factorB = Tensor<D COMMA T>({128 COMMA 8000}) COMMA
    }, T);

    // 2D (Non-Contiguous/Non-Contiguous) AL
    {
        Tensor<D, T> factorA({128, 1, 8000});
        Tensor<D, T> factorB({128, 1, 8000});

        factorA.view({{}, 0, {}});
        factorB.view({{}, 0, {}});

        JST_BENCHMARK_RUN("128x8000 * 128x8000 (2D-NC/NC) AL", {}, {
            .factorA = factorA COMMA
            .factorB = factorB COMMA
        }, T);
    }

    // Broadcast 2D (Contiguous/Non-Contiguous) 
    JST_BENCHMARK_RUN("128x8000 * 1x8000 (B-2D-C/NC)", {}, {
        .factorA = Tensor<D COMMA T>({128 COMMA 8000}) COMMA
        .factorB = Tensor<D COMMA T>({1 COMMA 8000}) COMMA
    }, T);

    // Broadcast 2D (Non-Contiguous/Non-Contiguous)
    {
        Tensor<D, T> factorA({128, 1, 8000});
        Tensor<D, T> factorB({1, 8000});

        factorA.view({{}, 0, {}});

        JST_BENCHMARK_RUN("128x8000 * 1x8000 (B-2D-NC/NC)", {}, {
            .factorA = factorA COMMA
            .factorB = factorB COMMA
        }, T);
    }

    // 3D (Contiguous/Contiguous) Alike
    JST_BENCHMARK_RUN("2x64x8000 * 2x64x8000 (3D-C/C) AL", {}, {
        .factorA = Tensor<D COMMA T>({2 COMMA 64 COMMA 8000}) COMMA
        .factorB = Tensor<D COMMA T>({2 COMMA 64 COMMA 8000}) COMMA
    }, T);

    // 3D (Non-Contiguous/Non-Contiguous) AL
    {
        Tensor<D, T> factorA({2, 1, 64, 8000});
        Tensor<D, T> factorB({2, 1, 64, 8000});

        factorA.view({{}, 0, {}, {}});
        factorB.view({{}, 0, {}, {}});

        JST_BENCHMARK_RUN("2x64x8000 * 2x64x8000 (3D-NC/NC) AL", {}, {
            .factorA = factorA COMMA
            .factorB = factorB COMMA
        }, T);
    }

    // Broadcast 3D (Contiguous/Non-Contiguous)
    JST_BENCHMARK_RUN("2x64x8000 * 2x1x8000 (B-3D-C/NC)", {}, {
        .factorA = Tensor<D COMMA T>({2 COMMA 64 COMMA 8000}) COMMA
        .factorB = Tensor<D COMMA T>({2 COMMA 1 COMMA 8000}) COMMA
    }, T);
}

}  // namespace Jetstream