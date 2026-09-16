#include "jetstream/viewport/adapters/generic.hh"

#include <iterator>
#include <utility>

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

Result Generic::addFileDropEvent(std::vector<std::string> paths) {
    std::lock_guard lock(droppedFilesMutex);
    droppedFiles.insert(droppedFiles.end(),
                        std::make_move_iterator(paths.begin()),
                        std::make_move_iterator(paths.end()));

    return Result::SUCCESS;
}

std::vector<std::string> Generic::takeDroppedFiles() {
    std::lock_guard lock(droppedFilesMutex);
    return std::exchange(droppedFiles, {});
}

}  // namespace Jetstream::Viewport 
