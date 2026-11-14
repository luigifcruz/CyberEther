#include <jetstream/render/sakura/components/spacer.hh>

#include <jetstream/render/sakura/components/stack.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>

namespace Jetstream::Sakura {

Spacer::Spacer() = default;
Spacer::~Spacer() = default;

void Spacer::update(Config next) {
    if (next == config) {
        return;
    }
    config = next;
    invalidate(Dirty::Layout);
}

Extent2D<F32> Spacer::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 lines = static_cast<F32>(std::max<U64>(1, config.lines) - 1);
    return {FillLength(available.x), FitLength(Px(ctx, Stack::Spacing) * lines, available.y)};
}

}  // namespace Jetstream::Sakura
