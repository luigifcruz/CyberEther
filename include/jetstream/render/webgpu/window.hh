#ifndef JETSTREAM_RENDER_WEBGPU_WINDOW_HH
#define JETSTREAM_RENDER_WEBGPU_WINDOW_HH

#include "jetstream/render/tools/imgui_impl_wgpu.h"

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/base.hh"

namespace Jetstream::Render {

template<>
class WindowImp<Device::WebGPU> : public Window {
 public:
    explicit WindowImp(const Config& config,
                       std::shared_ptr<Viewport::Adapter<Device::WebGPU>>& viewport);

    Result create() override;
    Result destroy() override;

    Result begin() override;
    Result end() override;

    const Stats& stats() const override;
    void drawDebugMessage() const override;

    constexpr Device device() const override {
        return Device::WebGPU;
    };

 protected:
    Result processSurfaceQueues() override;

 private:
    Stats statsData;
    ImGuiIO* io = nullptr;
    ImGuiStyle* style = nullptr;
    wgpu::CommandEncoder encoder;

    wgpu::RenderPassColorAttachment colorAttachments;
    wgpu::RenderPassDescriptor renderPassDesc;
    wgpu::Queue queue;

    std::vector<std::shared_ptr<SurfaceImp<Device::WebGPU>>> surfaces;
    std::shared_ptr<Viewport::Adapter<Device::WebGPU>> viewport;

    Result recreate();
    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
};

}  // namespace Jetstream::Render

#endif
