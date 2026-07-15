#ifndef JETSTREAM_RENDER_BASE_WINDOW_HH
#define JETSTREAM_RENDER_BASE_WINDOW_HH

#include <queue>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "jetstream/render/base/window_attachment.hh"
#include "jetstream/render/base/transfer.hh"
#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/parser.hh"
#include "jetstream/viewport/base.hh"
#include "jetstream/render/base/surface.hh"
#include "jetstream/render/components/generic.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/tools/imgui.h"
#include "jetstream/render/tools/imgui_stdlib.h"
#include "jetstream/render/tools/imgui_fmtlib.h"
#include "jetstream/render/tools/imgui_internal.h"

namespace Jetstream::Render::Components { class Font; }

namespace Jetstream::Render {

class JETSTREAM_API Window {
 public:
    struct Config {
        F32 scale = 1.0f;
    };

    struct Stats {
        U64 droppedFrames;
        U64 recreatedFrames;
    };

    explicit Window(const Config& config) : config(config) {}
    virtual ~Window() = default;

    const Config& getConfig() const {
        return config;
    }

    void setScale(F32 scale);

    Result create();
    Result destroy();

    Result start();
    Result stop();

    Result begin();
    Result end();
    Result cancel();

    Result synchronize();

    virtual const Stats& stats() const = 0;
    virtual std::string info() const = 0;

    virtual constexpr DeviceType device() const = 0;

    Result bind(const std::shared_ptr<Components::Generic>& component);
    Result unbind(const std::shared_ptr<Components::Generic>& component);

    Result bind(const std::shared_ptr<WindowAttachment>& attachment);
    Result unbind(const std::shared_ptr<WindowAttachment>& attachment);

    template<class T>
    inline Result JETSTREAM_API build(std::shared_ptr<T>& member,
                                      const auto& config) {
        // If the type is a component, create it.

        if constexpr (std::is_base_of_v<Components::Generic, T>) {
            member = std::make_shared<T>(config);
        }

        // If the type is not a component, create it based on the device.

        if constexpr (!std::is_base_of_v<Components::Generic, T>)  {
            switch (this->device()) {
    #ifdef JETSTREAM_RENDER_METAL_AVAILABLE
                case DeviceType::Metal:
                    member = T::template Factory<DeviceType::Metal>(config);
                    break;
    #endif
    #ifdef JETSTREAM_RENDER_VULKAN_AVAILABLE
                case DeviceType::Vulkan:
                    member = T::template Factory<DeviceType::Vulkan>(config);
                    break;
    #endif
    #ifdef JETSTREAM_RENDER_WEBGPU_AVAILABLE
                case DeviceType::WebGPU:
                    member = T::template Factory<DeviceType::WebGPU>(config);
                    break;
    #endif
                default:
                    JST_ERROR("Backend not supported yet.");
                    return Result::ERROR;
            }
        }

        return Result::SUCCESS;
    }

    constexpr const F32& scalingFactor() const {
        return _scalingFactor;
    }

    Extent2D<F32> framebufferScale() const;
    Extent2D<U64> framebufferSize(const Extent2D<F32>& displaySize) const;

    bool hasFont(const std::string& name) const;
    Result addFont(const std::string& name, const std::shared_ptr<Components::Font>& font);
    Result removeFont(const std::string& name);
    const std::shared_ptr<Components::Font>& font(const std::string& name) const;

 protected:
    Config config;

    virtual Result bindSurface(const std::shared_ptr<Surface>& surface) = 0;
    virtual Result unbindSurface(const std::shared_ptr<Surface>& surface) = 0;

    virtual Result underlyingCreate() = 0;
    virtual Result underlyingDestroy() = 0;

    virtual Result underlyingBegin() = 0;
    virtual Result underlyingEnd() = 0;
    virtual Result underlyingCancel() { return Result::SUCCESS; }

    virtual Result underlyingSynchronize() = 0;

    Result collectTransfers(Transfer::Batch& batch) const;
    void abortImguiFrame();

    // Font.

    std::unordered_map<std::string, std::shared_ptr<Components::Font>> fonts;

    // Scaling.

    F32 _scalingFactor;
    F32 _previousScalingFactor;

    void updateScalingFactor(const Viewport::Generic& viewport);

 private:
    bool shouldDeferAttachmentQueueProcessing();
    bool attachmentQueueEmpty() const;

    std::vector<std::shared_ptr<Components::Generic>> components;

    std::vector<std::function<void(const F32& scalingFactor)>> styleSetupCallbacks;
    std::vector<std::function<void(const F32& scalingFactor)>> styleScaleCallbacks;

    uint32_t frameCount;
    bool created = false;
    bool graphicalLoopThreadStarted;
    bool frameActive = false;
    std::thread::id graphicalLoopThreadId;
    std::mutex newFrameQueueMutex;

    // Attachment State

    struct PendingDestruction {
        uint32_t expiration;
        std::shared_ptr<WindowAttachment> attachment;
    };

    Result processAttachmentQueues();

    mutable std::mutex attachmentStateMutex;
    std::queue<std::shared_ptr<WindowAttachment>> bindQueue;
    std::queue<std::shared_ptr<WindowAttachment>> unbindQueue;
    std::queue<PendingDestruction> destroyQueue;

    std::vector<std::shared_ptr<WindowAttachment>> attachments;
    std::unordered_set<const WindowAttachment*> destroyingAttachments;
};

}  // namespace Jetstream::Render

#endif
