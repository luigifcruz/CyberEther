#include "jetstream/modules/multiply.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result Multiply<D, T>::create() {
    JST_DEBUG("Initializing Multiply module.");
    JST_INIT_IO();

    // Check parameters.

    const U64 rank_a = input.factorA.rank();
    const U64 rank_b = input.factorB.rank();

    const U64 min_rank = std::min(rank_a, rank_b);
    const U64 max_rank = std::max(rank_a, rank_b);

    const U64 pad_a = max_rank - rank_b;
    const U64 pad_b = max_rank - rank_a;

    JST_TRACE("[MULTIPLY] A rank: {}; B rank {};", rank_a, rank_b);
    JST_TRACE("[MULTIPLY] Min rank {}; Max rank {};", min_rank, max_rank);
    JST_TRACE("[MULTIPLY] Pad A {}; Pad B {};", pad_a, pad_b);

    for (U64 i = 0; i < min_rank; i++) {
        JST_TRACE("[MULTIPLY] Checking rank {} -> {} vs {}.", i, input.factorA.shape()[pad_a + i], 
                                                                 input.factorB.shape()[pad_b + i]);

        if (input.factorA.shape()[pad_a + i] != input.factorB.shape()[pad_b + i] && 
            input.factorA.shape()[pad_a + i] != 1 && 
            input.factorB.shape()[pad_b + i] != 1) {
            JST_ERROR("Input A {} and B {} shapes are not broadcastable.", input.factorA.shape(), 
                                                                           input.factorB.shape());
            return Result::ERROR;
        }
    }

    // Allocate output.

    std::vector<U64> output_shape(max_rank);

    for (U64 i = 0; i < max_rank; i++) {
        const U64 index_a = rank_a > i ? input.factorA.shape()[rank_a - 1 - i] : 1;
        const U64 index_b = rank_b > i ? input.factorB.shape()[rank_b - 1 - i] : 1;
        output_shape[max_rank - 1 - i] = std::max(index_a, index_b);
    }

    output.product = Tensor<D, T>(output_shape);

    return Result::SUCCESS;
}

template<Device D, typename T>
void Multiply<D, T>::info() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
