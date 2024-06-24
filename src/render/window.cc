#include "jetstream/render/base/window.hh"
#include "jetstream/platform.hh"

namespace Jetstream::Render {

Result Window::create() {
    // Set variables.

    _scalingFactor = 1.0f;
    _previousScalingFactor = 1.0f;
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

    // Call underlying destroy.
    const auto& res = underlyingDestroy();

    // Unlock the frame queue.
    newFrameQueueMutex.unlock();

    return res;
}

Result Window::begin() {
    // Process surface bind and unbind queue.
    JST_CHECK(processSurfaceUnbindQueue());
    JST_CHECK(processSurfaceBindQueue());

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

Result Window::bind(const std::shared_ptr<Surface>& surface) {
    // Push new surface to the bind queue.
    surfaceBindQueue.push(surface);

    // This is overcomplicated because of Emscripten.
    // The browser won't allow calling WebGPU function from other thread. 
    // So we need to find a way to make it work for everyone.

    // If graphical loop didn't start yet. Call the function directly.
    if (!graphicalLoopThreadStarted) {
        JST_CHECK(processSurfaceBindQueue());
    } 
    // Wait for graphical loop to process queue if current thread is different.
    else if (graphicalLoopThreadId != std::this_thread::get_id()) {
        while (!surfaceBindQueue.empty()) {
            std::this_thread::yield();
        }
    }
    // Call the function directly as fallback.
    else {
        JST_CHECK(processSurfaceBindQueue());
    }

    return Result::SUCCESS;
}

Result Window::unbind(const std::shared_ptr<Surface>& surface) {
    // Push new surface to the unbind queue.
    surfaceUnbindQueue.push(surface);

    // Wait completion.
    if (graphicalLoopThreadId != std::this_thread::get_id()) {
        while (!surfaceUnbindQueue.empty()) {
            std::this_thread::yield();
        }
    }
    // Call the function directly as fallback.
    else {
        JST_CHECK(processSurfaceUnbindQueue());
    }

    return Result::SUCCESS;
}

Result Window::processSurfaceBindQueue() {
    std::lock_guard<std::mutex> lock(newFrameQueueMutex);

    while (!surfaceBindQueue.empty()) {
        JST_CHECK(bindSurface(surfaceBindQueue.front()));
        surfaceBindQueue.pop();
    }

    return Result::SUCCESS;
}

Result Window::processSurfaceUnbindQueue() {
    std::lock_guard<std::mutex> lock(newFrameQueueMutex);

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

}  // namespace Jetstream::Render
