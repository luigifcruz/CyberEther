#ifndef JETSTREAM_RENDER_METAL_WINDOW_HH
#define JETSTREAM_RENDER_METAL_WINDOW_HH

#include "jetstream/render/tools/imgui_impl_metal.h"

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/base.hh"

namespace Jetstream::Render {

template<>
class WindowImp<Device::Metal> : public Window {
 public:
    explicit WindowImp(const Config& config,
                       std::shared_ptr<Viewport::Adapter<Device::Metal>>& viewport);

    Result create() override;
    Result destroy() override;

    Result begin() override;
    Result end() override;

    const Stats& stats() const override;
    void drawDebugMessage() const override;

    constexpr Device device() const override {
        return Device::Metal;
    };

    Result bind(const std::shared_ptr<Surface>& surface) override;
    Result unbind(const std::shared_ptr<Surface>& surface) override;

 private:
    Stats statsData;
    ImGuiIO* io = nullptr;
    ImGuiStyle* style = nullptr;
    MTL::Device* dev = nullptr;
    NS::AutoreleasePool* innerPool;
    NS::AutoreleasePool* outerPool;
    CA::MetalDrawable* drawable = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    MTL::CommandBuffer* commandBuffer = nullptr;
    MTL::RenderPassDescriptor* renderPassDescriptor = nullptr;

    std::vector<std::shared_ptr<SurfaceImp<Device::Metal>>> surfaces;
    std::shared_ptr<Viewport::Adapter<Device::Metal>> viewport;

    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
};

}  // namespace Jetstream::Render

#endif
