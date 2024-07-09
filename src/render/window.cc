#include "jetstream/render/components/font.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/platform.hh"

namespace Jetstream::Render {

Result Window::create() {
    // Set variables.

    _scalingFactor = 1.0f;
    _previousScalingFactor = 0.0f;
    graphicalLoopThreadStarted = false;

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

    // Lock the frame queue.
    newFrameQueueMutex.lock();

    // Destroy unclaimed components.

    for (const auto& component : components) {
        JST_WARN("[WINDOW] Destroying unclaimed component.");
        component->destroy(this);
    }

    // Call underlying destroy.
    const auto& res = underlyingDestroy();

    // Unlock the frame queue.
    newFrameQueueMutex.unlock();

    return res;
}

Result Window::begin() {
    // Process bind and unbind queue.
    JST_CHECK(processUnbindQueues());
    JST_CHECK(processBindQueues());

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

Result Window::bind(const std::shared_ptr<Components::Generic>& component) {
    // Call create on the component.
    JST_CHECK(component->create(this));

    // Push new component to the bind queue.
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

Result Window::bind(const std::shared_ptr<Buffer>& buffer) {
    // Push new buffer to the bind queue.
    bufferBindQueue.push(buffer);

    // Submit bind queues.
    JST_CHECK(submitBindQueues());

    return Result::SUCCESS;
}

Result Window::unbind(const std::shared_ptr<Buffer>& buffer) {
    // Push new buffer to the unbind queue.
    bufferUnbindQueue.push(buffer);

    // Submit unbind queues.
    JST_CHECK(submitUnbindQueues());

    return Result::SUCCESS;
}

Result Window::bind(const std::shared_ptr<Texture>& texture) {
    // Push new texture to the bind queue.
    textureBindQueue.push(texture);

    // Submit bind queues.
    JST_CHECK(submitBindQueues());

    return Result::SUCCESS;
}

Result Window::unbind(const std::shared_ptr<Texture>& texture) {
    // Push new texture to the unbind queue.
    textureUnbindQueue.push(texture);

    // Submit unbind queues.
    JST_CHECK(submitUnbindQueues());

    return Result::SUCCESS;
}

Result Window::bind(const std::shared_ptr<Surface>& surface) {
    // Push new surface to the bind queue.
    surfaceBindQueue.push(surface);

    // Submit bind queues.
    JST_CHECK(submitBindQueues());

    return Result::SUCCESS;
}

Result Window::unbind(const std::shared_ptr<Surface>& surface) {
    // Push new surface to the unbind queue.
    surfaceUnbindQueue.push(surface);

    // Submit unbind queues.
    JST_CHECK(submitUnbindQueues());

    return Result::SUCCESS;
}

Result Window::submitBindQueues() {
    // This is overcomplicated because of Emscripten.
    // The browser won't allow calling WebGPU function from other thread.
    // So we need to find a way to make it work for everyone.

    // If graphical loop didn't start yet. Call the function directly.
    if (!graphicalLoopThreadStarted) {
        JST_CHECK(processBindQueues());
    }
    // Wait for graphical loop to process queue if current thread is different.
    else if (graphicalLoopThreadId != std::this_thread::get_id()) {
        while (!surfaceBindQueue.empty() && !bufferBindQueue.empty() && !textureBindQueue.empty()) {
            std::this_thread::yield();
        }
    }
    // Call the function directly as fallback.
    else {
        JST_CHECK(processBindQueues());
    }

    return Result::SUCCESS;
}

Result Window::submitUnbindQueues() {
    // Wait completion.
    if (graphicalLoopThreadId != std::this_thread::get_id()) {
        while (!surfaceUnbindQueue.empty() && !bufferUnbindQueue.empty() && !textureUnbindQueue.empty()) {
            std::this_thread::yield();
        }
    }
    // Call the function directly as fallback.
    else {
        JST_CHECK(processUnbindQueues());
    }

    return Result::SUCCESS;
}


Result Window::processBindQueues() {
    std::lock_guard<std::mutex> lock(newFrameQueueMutex);

    while (!bufferBindQueue.empty()) {
        JST_CHECK(bindBuffer(bufferBindQueue.front()));
        bufferBindQueue.pop();
    }

    while (!textureBindQueue.empty()) {
        JST_CHECK(bindTexture(textureBindQueue.front()));
        textureBindQueue.pop();
    }

    while (!surfaceBindQueue.empty()) {
        JST_CHECK(bindSurface(surfaceBindQueue.front()));
        surfaceBindQueue.pop();
    }

    return Result::SUCCESS;
}

Result Window::processUnbindQueues() {
    std::lock_guard<std::mutex> lock(newFrameQueueMutex);

    while (!bufferUnbindQueue.empty()) {
        JST_CHECK(unbindBuffer(bufferUnbindQueue.front()));
        bufferUnbindQueue.pop();
    }

    while (!textureUnbindQueue.empty()) {
        JST_CHECK(unbindTexture(textureUnbindQueue.front()));
        textureUnbindQueue.pop();
    }

    while (!surfaceUnbindQueue.empty()) {
        JST_CHECK(unbindSurface(surfaceUnbindQueue.front()));
        surfaceUnbindQueue.pop();
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
