#include "jetstream/viewport/generic.hh"

namespace Jetstream::Viewport {

Generic::Generic(const Config& config) : config(config) {}

Result Generic::addMousePosEvent(F32 x, F32 y) {
    ImGui::GetIO().AddMousePosEvent(x, y);

    return Result::SUCCESS;
}

Result Generic::addMouseButtonEvent(U64 button, bool down) {
    ImGui::GetIO().AddMouseButtonEvent(button, down);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Viewport 
