#ifndef JETSTREAM_MEMORY2_HELPERS_HH
#define JETSTREAM_MEMORY2_HELPERS_HH

#include "jetstream/types.hh"
#include "jetstream/memory2/types.hh"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("unroll-loops")
#endif

namespace Jetstream::mem2 {

template<class Function, class... Args>
inline void AutomaticIterator(const Function& function, Args&... args) {
    const U64 size = std::max({args.size()...});
    const U64 rank = std::get<0>(std::forward_as_tuple(args...)).rank();

    const auto loop = [&]<class... Iterator>(Iterator&... iter) {
        std::array<U64, sizeof...(Args)> ptr = {args.offset()...};
        std::array<std::array<U64, 16>, sizeof...(Args)> coords = {};

        const std::array<const U64*, sizeof...(Args)> backstride = {args.backstride().data()...};
        const std::array<const U64*, sizeof...(Args)> shape_m1 = {args.shape_minus_one().data()...};
        const std::array<const U64*, sizeof...(Args)> stride = {args.stride().data()...};

        for (U64 i = 0; i < size; i++) {
            [&]<size_t... Is>(std::index_sequence<Is...>) __attribute__((always_inline)) {
                function(std::get<Is>(std::forward_as_tuple(args.data()...))[ptr[Is]]...);
            }(std::index_sequence_for<Args...>{});

            if constexpr (sizeof...(Iterator) == 1 && sizeof...(Args) > 1) {
                for (U64 x = 0; x < sizeof...(Args); x++) {
                    (iter(x, ptr, coords, stride, backstride, shape_m1), ...);
                }
            } else {
                [&]<size_t... Is>(std::index_sequence<Is...>) __attribute__((always_inline)) {
                    (std::get<Is>(std::forward_as_tuple(iter...))(Is, ptr, coords, stride, backstride, shape_m1), ...);
                }(std::index_sequence_for<Iterator...>{});
            }
        }
    };

    // TODO: If the strides and offsets are the same, iterate with a single counter.

    // Iterators 1D and Contiguous.

    const auto iterator_1d = [](const U64& i,
                                auto& ptr,
                                auto,
                                const auto& stride,
                                auto,
                                auto) __attribute__((always_inline)) {
        ptr[i] += stride[i][0];
    };

    const auto iterator_contiguous = [](const U64& i,
                                        auto& ptr,
                                        auto,
                                        auto,
                                        auto,
                                        auto) __attribute__((always_inline)) {
        ptr[i]++;
    };

    const auto iterator_2d = [](const U64& i,
                                auto& ptr,
                                auto& coords,
                                const auto& stride,
                                const auto& backstride,
                                const auto& shape_minus_one) __attribute__((always_inline)) {
        if (coords[i][1] < shape_minus_one[i][1]) [[likely]] {
            coords[i][1]++;
            ptr[i] += stride[i][1];
        } else [[unlikely]] {
            coords[i][1] = 0;
            coords[i][0]++;
            ptr[i] += stride[i][0] - backstride[i][1];
        }
    };

    const auto iterator_3d = [](const U64& i,
                                auto& ptr,
                                auto& coords,
                                const auto& stride,
                                const auto& backstride,
                                const auto& shape_minus_one) __attribute__((always_inline)) {
        for (I32 j = 2; j >= 0; j--) {
            if (coords[i][j] < shape_minus_one[i][j]) [[likely]] {
                coords[i][j] = coords[i][j] + 1;
                ptr[i] += stride[i][j];
                break;
            } else [[unlikely]] {
                coords[i][j] = 0;
                ptr[i] -= backstride[i][j];
            }
        }
    };

    // 1D

    if (rank == 1) {
        loop(iterator_1d);
        return;
    }

    // Contiguous

    if ((args.contiguous() && ...)) {
        loop(iterator_contiguous);
        return;
    }

    // 2D

    const std::array<bool, sizeof...(Args)> contiguous = {args.contiguous()...};

    if (rank == 2) {
        if constexpr (sizeof...(Args) == 3) {
            if (!contiguous[0] && !contiguous[1] && !contiguous[2]) {
                loop(iterator_2d);
                return;
            } else if (contiguous[0] && !contiguous[1] && !contiguous[2]) {
                loop(iterator_contiguous, iterator_2d, iterator_2d);
                return;
            } else if (!contiguous[0] && contiguous[1] && !contiguous[2]) {
                loop(iterator_2d, iterator_contiguous, iterator_2d);
                return;
            } else if (!contiguous[0] && !contiguous[1] && contiguous[2]) {
                loop(iterator_2d, iterator_2d, iterator_contiguous);
                return;
            } else if (contiguous[0] && contiguous[1] && !contiguous[2]) {
                loop(iterator_contiguous, iterator_contiguous, iterator_2d);
                return;
            } else if (contiguous[0] && !contiguous[1] && contiguous[2]) {
                loop(iterator_contiguous, iterator_2d, iterator_contiguous);
                return;
            } else if (!contiguous[0] && contiguous[1] && contiguous[2]) {
                loop(iterator_2d, iterator_contiguous, iterator_contiguous);
                return;
            }
        }

        if constexpr (sizeof...(Args) == 2) {
            if (!contiguous[0] && !contiguous[1]) {
                loop(iterator_2d);
                return;
            } else if (contiguous[0] && !contiguous[1]) {
                loop(iterator_contiguous, iterator_2d);
                return;
            } else if (!contiguous[0] && contiguous[1]) {
                loop(iterator_2d, iterator_contiguous);
                return;
            }
        }

        loop(iterator_2d);
        return;
    }

    // 3D

    if (rank == 3) {
        if constexpr (sizeof...(Args) == 3) {
            if (!contiguous[0] && !contiguous[1] && !contiguous[2]) {
                loop(iterator_3d);
                return;
            } else if (contiguous[0] && !contiguous[1] && !contiguous[2]) {
                loop(iterator_contiguous, iterator_3d, iterator_3d);
                return;
            } else if (!contiguous[0] && contiguous[1] && !contiguous[2]) {
                loop(iterator_3d, iterator_contiguous, iterator_3d);
                return;
            } else if (!contiguous[0] && !contiguous[1] && contiguous[2]) {
                loop(iterator_3d, iterator_3d, iterator_contiguous);
                return;
            } else if (contiguous[0] && contiguous[1] && !contiguous[2]) {
                loop(iterator_contiguous, iterator_contiguous, iterator_3d);
                return;
            } else if (contiguous[0] && !contiguous[1] && contiguous[2]) {
                loop(iterator_contiguous, iterator_3d, iterator_contiguous);
                return;
            } else if (!contiguous[0] && contiguous[1] && contiguous[2]) {
                loop(iterator_3d, iterator_contiguous, iterator_contiguous);
                return;
            }
        }

        if constexpr (sizeof...(Args) == 2) {
            if (!contiguous[0] && !contiguous[1]) {
                loop(iterator_3d);
                return;
            } else if (contiguous[0] && !contiguous[1]) {
                loop(iterator_contiguous, iterator_3d);
                return;
            } else if (!contiguous[0] && contiguous[1]) {
                loop(iterator_3d, iterator_contiguous);
                return;
            }
        }

        loop(iterator_3d);
        return;
    }

    JST_FATAL("Automatic iterator not implemented for rank {}.", rank);
    JST_CHECK_THROW(Result::FATAL);
}

}  // namespace Jetstream::mem2

#endif
