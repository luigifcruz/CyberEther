#include "jetstream/modules/add.hh"

namespace Jetstream {

template<Device D, typename T>
Result Add<D, T>::create() {
    JST_DEBUG("Initializing Add module.");
    JST_INIT_IO();

    // Check parameters.

    const U64 rank_a = input.addendA.rank();
    const U64 rank_b = input.addendB.rank();

    const U64 min_rank = std::min(rank_a, rank_b);
    const U64 max_rank = std::max(rank_a, rank_b);

    const U64 pad_a = max_rank - rank_b;
    const U64 pad_b = max_rank - rank_a;

    JST_TRACE("[ADD] A rank: {}; B rank {};", rank_a, rank_b);
    JST_TRACE("[ADD] Min rank {}; Max rank {};", min_rank, max_rank);
    JST_TRACE("[ADD] Pad A {}; Pad B {};", pad_a, pad_b);

    for (U64 i = 0; i < min_rank; i++) {
        JST_TRACE("[ADD] Checking rank {} -> {} vs {}.", i, input.addendA.shape()[pad_a + i],
                                                             input.addendB.shape()[pad_b + i]);

        if (input.addendA.shape()[pad_a + i] != input.addendB.shape()[pad_b + i] &&
            input.addendA.shape()[pad_a + i] != 1 &&
            input.addendB.shape()[pad_b + i] != 1) {
            JST_ERROR("Input A {} and B {} shapes are not broadcastable.", input.addendA.shape(),
                                                                           input.addendB.shape());
            return Result::ERROR;
        }
    }

    // Allocate output.

    mem2::Shape output_shape(max_rank);

    for (U64 i = 0; i < max_rank; i++) {
        const U64 index_a = rank_a > i ? input.addendA.shape()[rank_a - 1 - i] : 1;
        const U64 index_b = rank_b > i ? input.addendB.shape()[rank_b - 1 - i] : 1;
        output_shape[max_rank - 1 - i] = std::max(index_a, index_b);
    }

    JST_CHECK(output.sum.create(D, mem2::TypeToDataType<T>(), output_shape));

    // Broadcast input.

    JST_CHECK(impl->a.create(D, input.addendA));
    JST_CHECK(impl->b.create(D, input.addendB));
    JST_CHECK(impl->c.create(D, output.sum));

    JST_CHECK(impl->a.broadcast_to(impl->c.shape()));
    JST_CHECK(impl->b.broadcast_to(impl->c.shape()));

    return Result::SUCCESS;
}

template<Device D, typename T>
void Add<D, T>::info() const {
    JST_DEBUG("  None");
}

}  // namespace Jetstream