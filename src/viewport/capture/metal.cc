#include "jetstream/viewport/capture/metal.hh"
#include "jetstream/viewport/adapters/metal.hh"

#include <array>
#include <condition_variable>
#include <mutex>
#include <queue>

namespace Jetstream::Viewport {

struct FrameCaptureMetal::Impl {
    static constexpr U32 MAX_FRAMES_IN_FLIGHT = 2;

    Adapter<DeviceType::Metal>* metalViewport = nullptr;
    Extent2D<U64> extent = {};

    std::array<Tensor, MAX_FRAMES_IN_FLIGHT> stagingTensors = {};
    std::array<Tensor, MAX_FRAMES_IN_FLIGHT> outputTensors = {};
    std::array<bool, MAX_FRAMES_IN_FLIGHT> frameEvents = {};

    std::queue<U32> frameQueue;
    std::mutex frameMutex;
    std::condition_variable frameCond;
    bool running = false;
    bool requested = false;

    U32 currentCaptureIndex = 0;
    U32 currentReadyIndex = MAX_FRAMES_IN_FLIGHT;
};

FrameCaptureMetal::FrameCaptureMetal() = default;

FrameCaptureMetal::~FrameCaptureMetal() {
    destroy();
}

Result FrameCaptureMetal::create(Generic* viewport, const DeviceType& outputDevice) {
    JST_DEBUG("[CAPTURE] Creating Metal frame capture.");

    JST_CHECK(destroy());

    auto* adapter = dynamic_cast<Adapter<DeviceType::Metal>*>(viewport);
    if (!adapter || outputDevice != DeviceType::CPU) {
        JST_ERROR("[CAPTURE] Metal frame capture requires a Metal viewport and CPU output.");
        return Result::ERROR;
    }

    auto state = std::make_shared<Impl>();
    state->extent = adapter->getSwapchainExtent();
    if (state->extent.x == 0 || state->extent.y == 0) {
        JST_ERROR("[CAPTURE] Metal viewport has an empty extent.");
        return Result::ERROR;
    }

    for (U32 i = 0; i < Impl::MAX_FRAMES_IN_FLIGHT; i++) {
        const Shape extent = {state->extent.y, state->extent.x, 4};
        const Buffer::Config config = {.hostAccessible = true};
        JST_CHECK(state->stagingTensors[i].create(DeviceType::Metal, DataType::U8, extent, config));
        if (state->outputTensors[i].create(outputDevice, state->stagingTensors[i]) != Result::SUCCESS) {
            JST_ERROR("[CAPTURE] Staging tensor does not support requested output device.");
            return Result::ERROR;
        }
    }

    state->metalViewport = adapter;
    state->running = true;
    pimpl = state;

    adapter->setCaptureCallback([state](MTL::CommandBuffer* commandBuffer, MTL::Texture* texture) {
        std::lock_guard lock(state->frameMutex);
        if (!state->running || !state->requested) {
            return Result::SUCCESS;
        }
        state->requested = false;

        const U32 index = state->currentCaptureIndex;
        if (state->frameEvents[index]) {
            return Result::SUCCESS;
        }

        if (texture->pixelFormat() != MTL::PixelFormatBGRA8Unorm) {
            JST_ERROR("[CAPTURE] Metal frame capture requires BGRA8Unorm textures.");
            return Result::ERROR;
        }
        if (texture->width() < state->extent.x || texture->height() < state->extent.y) {
            return Result::SUCCESS;
        }

        auto* encoder = commandBuffer->blitCommandEncoder();
        if (!encoder) {
            JST_ERROR("[CAPTURE] Failed to create Metal blit encoder.");
            return Result::ERROR;
        }
        auto* buffer = static_cast<MTL::Buffer*>(state->stagingTensors[index].data());
        const U64 rowBytes = state->extent.x * 4;
        encoder->copyFromTexture(texture, 0, 0, MTL::Origin(0, 0, 0),
                                 MTL::Size(state->extent.x, state->extent.y, 1), buffer,
                                 0, rowBytes, rowBytes * state->extent.y);
        encoder->endEncoding();
        state->frameEvents[index] = true;

        commandBuffer->addCompletedHandler([state, index](MTL::CommandBuffer* completed) {
            std::lock_guard lock(state->frameMutex);
            if (state->running && completed->status() == MTL::CommandBufferStatusCompleted) {
                state->frameQueue.push(index);
                state->frameCond.notify_one();
            } else {
                state->frameEvents[index] = false;
            }
        });

        state->currentCaptureIndex = (index + 1) % Impl::MAX_FRAMES_IN_FLIGHT;
        return Result::SUCCESS;
    });

    JST_DEBUG("[CAPTURE] Metal frame capture created ({}x{}).", state->extent.x, state->extent.y);
    return Result::SUCCESS;
}

Result FrameCaptureMetal::stop() {
    JST_DEBUG("[CAPTURE] Stopping Metal frame capture.");

    if (pimpl) {
        std::lock_guard lock(pimpl->frameMutex);
        pimpl->running = false;
        pimpl->requested = false;
        pimpl->frameCond.notify_all();
    }
    return Result::SUCCESS;
}

Result FrameCaptureMetal::destroy() {
    JST_DEBUG("[CAPTURE] Destroying Metal frame capture.");

    JST_CHECK(stop());
    if (pimpl && pimpl->metalViewport) {
        pimpl->metalViewport->setCaptureCallback({});
    }
    pimpl.reset();
    return Result::SUCCESS;
}

Result FrameCaptureMetal::captureFrame() {
    if (pimpl) {
        std::lock_guard lock(pimpl->frameMutex);
        pimpl->requested = pimpl->running;
    }
    return Result::SUCCESS;
}

Result FrameCaptureMetal::getFrameData(Tensor& tensor) {
    const auto state = pimpl;
    if (!state) {
        return Result::ERROR;
    }

    std::unique_lock lock(state->frameMutex);
    if (state->currentReadyIndex != Impl::MAX_FRAMES_IN_FLIGHT) {
        return Result::ERROR;
    }
    state->frameCond.wait(lock, [&] { return !state->running || !state->frameQueue.empty(); });
    if (!state->running) {
        return Result::ERROR;
    }

    state->currentReadyIndex = state->frameQueue.front();
    state->frameQueue.pop();
    tensor = state->outputTensors[state->currentReadyIndex];
    return Result::SUCCESS;
}

Result FrameCaptureMetal::releaseFrame() {
    const auto state = pimpl;
    if (state) {
        std::lock_guard lock(state->frameMutex);
        if (state->currentReadyIndex != Impl::MAX_FRAMES_IN_FLIGHT) {
            state->frameEvents[state->currentReadyIndex] = false;
            state->currentReadyIndex = Impl::MAX_FRAMES_IN_FLIGHT;
        }
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream::Viewport
