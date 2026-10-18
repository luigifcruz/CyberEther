#include "jetstream/viewport/adapters/generic.hh"

#include <iterator>
#include <utility>

namespace Jetstream::Viewport {

Generic::Generic(const Config& config) : config(config) {}

Extent2D<F32> Generic::displaySize() const {
    return {static_cast<F32>(config.size.x), static_cast<F32>(config.size.y)};
}

Render::TitlebarChrome Generic::titlebarChrome() const {
    return {};
}

void Generic::setIntegratedTitlebar(bool enabled) {
    config.integratedTitlebar = enabled;
}

bool Generic::integratedTitlebar() const {
    return config.integratedTitlebar;
}

void Generic::beginWindowDrag() {}

void Generic::zoomWindow() {}

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
