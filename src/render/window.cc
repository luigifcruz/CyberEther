#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window_attachment.hh"
#include "jetstream/render/components/font.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/platform.hh"

namespace Jetstream::Render {

Result Window::create() {
    // Set variables.

    _scalingFactor = 1.0f;
    _previousScalingFactor = 0.0f;
    graphicalLoopThreadStarted = false;
    frameCount = 0;

    // Lock the frame queue.
    newFrameQueueMutex.lock();

    // Call underlying create.
    const auto& res = underlyingCreate();

    // Unlock the frame queue.
    newFrameQueueMutex.unlock();

    return res;
}

Result Window::destroy() {
    graphicalLoopThreadStarted = false;

    // Unbind unclaimed components.

    while (!components.empty()) {
        JST_WARN("[WINDOW] Destroying unclaimed component.");
        JST_CHECK(unbind(components.front()));
    }
    JST_CHECK(processAttachmentQueues());

    // Destroy unclaimed attachments.

    while (!attachments.empty()) {
        JST_WARN("[WINDOW] Destroying unclaimed attachment.");
        JST_CHECK(unbind(attachments.front()));
    }
    JST_CHECK(processAttachmentQueues());

    // Lock the frame queue.
    newFrameQueueMutex.lock();

    // Call underlying destroy.
    const auto& res = underlyingDestroy();

    // Process pending attachment destruction.

    while (!destroyQueue.empty()) {
        auto& [_, attachment] = destroyQueue.front();
        destroyQueue.pop();

        JST_CHECK(attachment->destroy());
    }

    // Unlock the frame queue.
    newFrameQueueMutex.unlock();

    return res;
}

Result Window::begin() {
    // Process attachment queues.
    JST_CHECK(processAttachmentQueues());

    // Record graphical thread ID.

    graphicalLoopThreadStarted = true;
    graphicalLoopThreadId = std::this_thread::get_id();

    // Lock the frame queue.
    newFrameQueueMutex.lock();

    // Call frame begin.
    const auto& res = underlyingBegin();

    // Unlock the frame queue if failed.

    if (res != Result::SUCCESS) {
        newFrameQueueMutex.unlock();
    }

    return res;
}

Result Window::end() {
    // Call frame end.
    const auto& res = underlyingEnd();

    // Process attachment destruction queue.

    int count = destroyQueue.size();
    while (count > 0) {
        auto& [expiration, attachment] = destroyQueue.front();

        if (expiration <= frameCount) {
            JST_CHECK(attachment->destroy());
            destroyQueue.pop();
        }

        count--;
    }

    // Increment frame count.
    frameCount++;

    // Unlock the frame queue.
    newFrameQueueMutex.unlock();

    return res;
}

Result Window::synchronize() {
    // Lock the frame queue.
    newFrameQueueMutex.lock();

    // Call frame synchronize.
    const auto& res = underlyingSynchronize();

    // Unlock the frame queue.
    newFrameQueueMutex.unlock();

    return res;
}

Result Window::bind(const std::shared_ptr<WindowAttachment>& attachment) {
    // Add attachment to bind queue.
    bindQueue.push(attachment);

    // Wait queue to be processed.

    if (graphicalLoopThreadStarted && (graphicalLoopThreadId != std::this_thread::get_id())) {
        while (!bindQueue.empty()) {
            std::this_thread::yield();
        }
    } else {
        processAttachmentQueues();
    }

    // Push attachment to the list.
    attachments.push_back(attachment);

    return Result::SUCCESS;
}

Result Window::unbind(const std::shared_ptr<WindowAttachment>& attachment) {
    // Add attachment to unbind queue.
    unbindQueue.push(attachment);

    // Remove component from the list.
    attachments.erase(std::remove(attachments.begin(), attachments.end(), attachment), attachments.end());

    return Result::SUCCESS;
}

Result Window::processAttachmentQueues() {
    std::lock_guard<std::mutex> lock(newFrameQueueMutex);

    // Process unbind attachment queue.

    while (!unbindQueue.empty()) {
        auto attachment = unbindQueue.front();
        unbindQueue.pop();

        // Schedule for destruction.

        destroyQueue.push({
            .expiration = frameCount + 4,  // TODO: Replace hardcoded value with implementation-specific value.
            .attachment = attachment
        });

        // Unregister Surface.

        if (attachment->type() == WindowAttachment::Type::Surface) {
            JST_CHECK(unbindSurface(std::dynamic_pointer_cast<Surface>(attachment)));
        }
    }

    // Process bind attachment queue.

    std::vector<std::shared_ptr<Surface>> surfaces;

    while (!bindQueue.empty()) {
        auto attachment = bindQueue.front();
        bindQueue.pop();

        // Create attachment
        JST_CHECK(attachment->create());

        // Schedule Surface registration.

        if (attachment->type() == WindowAttachment::Type::Surface) {
            surfaces.push_back(std::dynamic_pointer_cast<Surface>(attachment));
        }
    }

    // Register Surfaces.

    for (auto& surface : surfaces) {
        JST_CHECK(bindSurface(surface));
    }

    return Result::SUCCESS;
}

void Window::scaleStyle(const Viewport::Generic& viewport) {
    _scalingFactor = viewport.scale(config.scale);

    if (_scalingFactor != _previousScalingFactor) {
        auto& style = ImGui::GetStyle();
        style.ScaleAllSizes(_scalingFactor);
    }

    _previousScalingFactor = _scalingFactor;
}

Result Window::bind(const std::shared_ptr<Components::Generic>& component) {
    // Call create on the component.
    JST_CHECK(component->create(this));

    // Push component to the list.
    components.push_back(component);

    return Result::SUCCESS;
}

Result Window::unbind(const std::shared_ptr<Components::Generic>& component) {
    // Call destroy on the component.
    JST_CHECK(component->destroy(this));

    // Remove component from the list.
    components.erase(std::remove(components.begin(), components.end(), component), components.end());

    return Result::SUCCESS;
}

bool Window::hasFont(const std::string& name) const {
    return fonts.contains(name);
}

Result Window::addFont(const std::string& name, const std::shared_ptr<Components::Font>& font) {
    // Check if font exists.

    if (hasFont(name)) {
        JST_ERROR("[WINDOW] Font '{}' already exists.", name);
        return Result::ERROR;
    }

    // Bind to window.

    JST_CHECK(this->bind(font));

    // Add to list.

    fonts[name] = font;

    return Result::SUCCESS;
}

Result Window::removeFont(const std::string& name) {
    // Check if font exists.

    if (!hasFont(name)) {
        JST_ERROR("[WINDOW] Font '{}' does not exist.", name);
        return Result::ERROR;
    }

    // Unbind from window.

    const auto& font = fonts.at(name);
    JST_CHECK(this->unbind(font));

    // Remove from list.

    fonts.erase(name);

    return Result::SUCCESS;
}

const std::shared_ptr<Components::Font>& Window::font(const std::string& name) const {
    return fonts.at(name);
}

}  // namespace Jetstream::Render
