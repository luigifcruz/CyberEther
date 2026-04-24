#ifndef JETSTREAM_TOOLS_AUTOMATIC_ITERATOR_HH
#define JETSTREAM_TOOLS_AUTOMATIC_ITERATOR_HH

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/memory/types.hh"
#include "jetstream/memory/tensor.hh"

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("unroll-loops")
#endif

namespace Jetstream {

namespace detail {

template<typename Function, typename DataTuple, std::size_t N, std::size_t... Is>
JST_INLINE void AutomaticIteratorInvoke(const Function& function,
                                        const DataTuple& dataPtrs,
                                        const std::array<U64, N>& ptr,
                                        std::index_sequence<Is...>) {
    function((std::get<Is>(dataPtrs)[ptr[Is]])...);
}

template<typename IteratorTuple, std::size_t N, std::size_t... Is>
JST_INLINE void AutomaticIteratorStep(IteratorTuple&& iterators,
                                      std::array<U64, N>& ptr,
                                      std::array<std::array<U64, 16>, N>& coords,
                                      const std::array<const U64*, N>& stride,
                                      const std::array<const U64*, N>& backstride,
                                      const std::array<const U64*, N>& shapeM1,
                                      std::index_sequence<Is...>) {
    (std::get<Is>(iterators)(Is, ptr, coords, stride, backstride, shapeM1), ...);
}

struct AutomaticIterator1D {
    template<typename PtrArray, typename CoordsArray, typename StrideArray, typename BackstrideArray, typename ShapeMinusOneArray>
    JST_INLINE void operator()(const U64& i,
                               PtrArray& ptr,
                               CoordsArray&,
                               const StrideArray& stride,
                               const BackstrideArray&,
                               const ShapeMinusOneArray&) const {
        ptr[i] += stride[i][0];
    }
};

struct AutomaticIteratorContiguous {
    template<typename PtrArray, typename CoordsArray, typename StrideArray, typename BackstrideArray, typename ShapeMinusOneArray>
    JST_INLINE void operator()(const U64& i,
                               PtrArray& ptr,
                               CoordsArray&,
                               const StrideArray&,
                               const BackstrideArray&,
                               const ShapeMinusOneArray&) const {
        ptr[i]++;
    }
};

struct AutomaticIterator2D {
    template<typename PtrArray, typename CoordsArray, typename StrideArray, typename BackstrideArray, typename ShapeMinusOneArray>
    JST_INLINE void operator()(const U64& i,
                               PtrArray& ptr,
                               CoordsArray& coords,
                               const StrideArray& stride,
                               const BackstrideArray& backstride,
                               const ShapeMinusOneArray& shapeMinusOne) const {
        if (coords[i][1] < shapeMinusOne[i][1]) [[likely]] {
            coords[i][1]++;
            ptr[i] += stride[i][1];
        } else [[unlikely]] {
            coords[i][1] = 0;
            coords[i][0]++;
            ptr[i] += stride[i][0] - backstride[i][1];
        }
    }
};

struct AutomaticIterator3D {
    template<typename PtrArray, typename CoordsArray, typename StrideArray, typename BackstrideArray, typename ShapeMinusOneArray>
    JST_INLINE void operator()(const U64& i,
                               PtrArray& ptr,
                               CoordsArray& coords,
                               const StrideArray& stride,
                               const BackstrideArray& backstride,
                               const ShapeMinusOneArray& shapeMinusOne) const {
        for (I32 j = 2; j >= 0; j--) {
            if (coords[i][j] < shapeMinusOne[i][j]) [[likely]] {
                coords[i][j] = coords[i][j] + 1;
                ptr[i] += stride[i][j];
                break;
            } else [[unlikely]] {
                coords[i][j] = 0;
                ptr[i] -= backstride[i][j];
            }
        }
    }
};

}  // namespace detail

template<typename... Elem, class Function, class... Args>
JST_INLINE Result AutomaticIterator(const Function& function, Args&... args) {
    static_assert(sizeof...(Elem) == sizeof...(Args),
                  "Number of element types must match number of tensor arguments.");

    if (!((args.device() == DeviceType::CPU) && ...)) {
        JST_ERROR("AutomaticIterator only supports CPU tensors.");
        return Result::ERROR;
    }

    constexpr size_t N = sizeof...(Args);

    const U64 size = std::max({args.size()...});
    const Index rank = std::get<0>(std::forward_as_tuple(args...)).rank();

    // Pre-cast data pointers to typed pointers
    auto dataPtrs = std::tuple<std::add_pointer_t<std::remove_reference_t<Elem>>...>{
        static_cast<std::add_pointer_t<std::remove_reference_t<Elem>>>(args.data())...
    };

    const auto loop = [&]<class... Iterator>(Iterator&... iter) {
        std::array<U64, N> ptr = {};
        std::array<std::array<U64, 16>, N> coords = {};

        const std::array<const U64*, N> backstride = {args.backstride().data()...};
        const std::array<const U64*, N> shapeM1   = {args.shapeMinusOne().data()...};
        const std::array<const U64*, N> stride     = {args.stride().data()...};

        for (U64 i = 0; i < size; i++) {
            detail::AutomaticIteratorInvoke(function, dataPtrs, ptr, std::make_index_sequence<N>{});

            if constexpr (sizeof...(Iterator) == 1 && N > 1) {
                for (U64 x = 0; x < N; x++) {
                    (iter(x, ptr, coords, stride, backstride, shapeM1), ...);
                }
            } else {
                detail::AutomaticIteratorStep(std::forward_as_tuple(iter...),
                                              ptr,
                                              coords,
                                              stride,
                                              backstride,
                                              shapeM1,
                                              std::make_index_sequence<sizeof...(Iterator)>{});
            }
        }
    };

    // Iterator implementations (same as main function)
    constexpr detail::AutomaticIterator1D iterator1d{};
    constexpr detail::AutomaticIteratorContiguous iteratorContiguous{};
    constexpr detail::AutomaticIterator2D iterator2d{};
    constexpr detail::AutomaticIterator3D iterator3d{};

    // 1D
    if (rank == 1) {
        loop(iterator1d);
        return Result::SUCCESS;
    }

    // Contiguous (all)
    if ((args.contiguous() && ...)) {
        loop(iteratorContiguous);
        return Result::SUCCESS;
    }

    // 2D
    const std::array<bool, N> contiguous = {args.contiguous()...};

    if (rank == 2) {
        if constexpr (N == 3) {
            if (!contiguous[0] && !contiguous[1] && !contiguous[2]) {
                loop(iterator2d);
                return Result::SUCCESS;
            } else if (contiguous[0] && !contiguous[1] && !contiguous[2]) {
                loop(iteratorContiguous, iterator2d, iterator2d);
                return Result::SUCCESS;
            } else if (!contiguous[0] && contiguous[1] && !contiguous[2]) {
                loop(iterator2d, iteratorContiguous, iterator2d);
                return Result::SUCCESS;
            } else if (!contiguous[0] && !contiguous[1] && contiguous[2]) {
                loop(iterator2d, iterator2d, iteratorContiguous);
                return Result::SUCCESS;
            } else if (contiguous[0] && contiguous[1] && !contiguous[2]) {
                loop(iteratorContiguous, iteratorContiguous, iterator2d);
                return Result::SUCCESS;
            } else if (contiguous[0] && !contiguous[1] && contiguous[2]) {
                loop(iteratorContiguous, iterator2d, iteratorContiguous);
                return Result::SUCCESS;
            } else if (!contiguous[0] && contiguous[1] && contiguous[2]) {
                loop(iterator2d, iteratorContiguous, iteratorContiguous);
                return Result::SUCCESS;
            }
        }

        if constexpr (N == 2) {
            if (!contiguous[0] && !contiguous[1]) {
                loop(iterator2d);
                return Result::SUCCESS;
            } else if (contiguous[0] && !contiguous[1]) {
                loop(iteratorContiguous, iterator2d);
                return Result::SUCCESS;
            } else if (!contiguous[0] && contiguous[1]) {
                loop(iterator2d, iteratorContiguous);
                return Result::SUCCESS;
            }
        }

        loop(iterator2d);
        return Result::SUCCESS;
    }

    // 3D
    if (rank == 3) {
        if constexpr (N == 3) {
            if (!contiguous[0] && !contiguous[1] && !contiguous[2]) {
                loop(iterator3d);
                return Result::SUCCESS;
            } else if (contiguous[0] && !contiguous[1] && !contiguous[2]) {
                loop(iteratorContiguous, iterator3d, iterator3d);
                return Result::SUCCESS;
            } else if (!contiguous[0] && contiguous[1] && !contiguous[2]) {
                loop(iterator3d, iteratorContiguous, iterator3d);
                return Result::SUCCESS;
            } else if (!contiguous[0] && !contiguous[1] && contiguous[2]) {
                loop(iterator3d, iterator3d, iteratorContiguous);
                return Result::SUCCESS;
            } else if (contiguous[0] && contiguous[1] && !contiguous[2]) {
                loop(iteratorContiguous, iteratorContiguous, iterator3d);
                return Result::SUCCESS;
            } else if (contiguous[0] && !contiguous[1] && contiguous[2]) {
                loop(iteratorContiguous, iterator3d, iteratorContiguous);
                return Result::SUCCESS;
            } else if (!contiguous[0] && contiguous[1] && contiguous[2]) {
                loop(iterator3d, iteratorContiguous, iteratorContiguous);
                return Result::SUCCESS;
            }
        }

        if constexpr (N == 2) {
            if (!contiguous[0] && !contiguous[1]) {
                loop(iterator3d);
                return Result::SUCCESS;
            } else if (contiguous[0] && !contiguous[1]) {
                loop(iteratorContiguous, iterator3d);
                return Result::SUCCESS;
            } else if (!contiguous[0] && contiguous[1]) {
                loop(iterator3d, iteratorContiguous);
                return Result::SUCCESS;
            }
        }

        loop(iterator3d);
        return Result::SUCCESS;
    }

    JST_ERROR("Automatic iterator not implemented for rank {}.", rank);
    return Result::ERROR;
}

}  // namespace Jetstream

#endif  // JETSTREAM_TOOLS_AUTOMATIC_ITERATOR_HH
