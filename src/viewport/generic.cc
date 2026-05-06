#include "jetstream/viewport/adapters/generic.hh"

namespace Jetstream::Viewport {

Generic::Generic(const Config& config) : config(config) {}

Extent2D<F32> Generic::displaySize() const {
    return {static_cast<F32>(config.size.x), static_cast<F32>(config.size.y)};
}

Result Generic::addMousePosEvent(F32 x, F32 y) {
    ImGui::GetIO().AddMousePosEvent(x, y);

    return Result::SUCCESS;
}

Result Generic::addMouseButtonEvent(U64 button, bool down) {
    ImGui::GetIO().AddMouseButtonEvent(button, down);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Viewport 
