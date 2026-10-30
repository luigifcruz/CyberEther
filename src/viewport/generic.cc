#include "jetstream/viewport/adapters/generic.hh"

#include <chrono>
#include <iterator>
#include <utility>

namespace Jetstream::Viewport {

Generic::Generic(const Config& config) : config(config) {}

Generic::~Generic() = default;

Extent2D<F32> Generic::displaySize() const {
    return {static_cast<F32>(config.size.x), static_cast<F32>(config.size.y)};
}

Extent2D<F32> Generic::framebufferSize() const {
    return displaySize();
}

Extent2D<F32> Generic::contentScale() const {
    return {1.0f, 1.0f};
}

Render::WindowMetrics::Policy Generic::scalePolicy() const {
    return Render::WindowMetrics::Policy::Interface;
}

Render::WindowMetrics Generic::metrics(F32 interfaceScale) const {
    {
        std::lock_guard lock(snapshotMutex);
        if (snapshot.has_value()) {
            return Render::WindowMetrics::Compute(snapshot->displaySize, snapshot->framebufferSize,
                                                  snapshot->contentScale, interfaceScale, scalePolicy());
        }
    }
    return Render::WindowMetrics::Compute(displaySize(), framebufferSize(), contentScale(), interfaceScale, scalePolicy());
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

F64 Generic::inputTime() const {
    if (platform) {
        return platform->time();
    }
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration<F64>(now).count();
}

bool Generic::macOSBehaviors() const {
#if defined(JST_OS_MAC) || defined(JST_OS_IOS)
    return true;
#else
    return false;
#endif
}

bool Generic::retainedPresent() const {
    return false;
}

void Generic::setCursor(SurfaceCursor cursor) {
    if (platform) {
        platform->setCursor(cursor);
    }
}

std::string Generic::clipboardText() const {
    return platform ? platform->clipboardText() : clipboard;
}

void Generic::setClipboardText(const std::string& text) {
    if (platform) {
        platform->setClipboardText(text);
        return;
    }
    clipboard = text;
}

void Generic::attachPlatform(std::unique_ptr<Platform> next) {
    platform = std::move(next);
}

void Generic::detachPlatform() {
    platform.reset();
    std::lock_guard lock(snapshotMutex);
    snapshot.reset();
}

void Generic::pumpPlatform() {
    if (!platform) {
        return;
    }
    platform->poll();
    const Snapshot next = {displaySize(), framebufferSize(), contentScale()};
    std::lock_guard lock(snapshotMutex);
    snapshot = next;
}

void Generic::waitIdle(F64 seconds) {
    inputQueue.wait(seconds);
}

void Generic::requestRedraw() {
    redrawRequested = true;
    inputQueue.wake();
}

bool Generic::consumeRedraw() {
    return redrawRequested.exchange(false);
}

Render::InputQueue& Generic::input() {
    return inputQueue;
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
