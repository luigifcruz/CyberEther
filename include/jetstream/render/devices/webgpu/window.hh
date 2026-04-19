#ifndef JETSTREAM_RENDER_WEBGPU_WINDOW_HH
#define JETSTREAM_RENDER_WEBGPU_WINDOW_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/render/tools/imgui.h"
#include "jetstream/viewport/base.hh"

#include <webgpu/webgpu.h>

namespace Jetstream::Render {

template<>
class WindowImp<DeviceType::WebGPU> : public Window {
 public:
    explicit WindowImp(const Config& config,
                       const std::shared_ptr<Viewport::Adapter<DeviceType::WebGPU>>& viewport);

    const Stats& stats() const override;
    std::string info() const override;

    constexpr DeviceType device() const override {
        return DeviceType::WebGPU;
    };

 protected:
    Result bindSurface(const std::shared_ptr<Surface>& surface) override;
    Result unbindSurface(const std::shared_ptr<Surface>& surface) override;

    Result underlyingCreate() override;
    Result underlyingDestroy() override;

    Result underlyingBegin() override;
    Result underlyingEnd() override;

    Result underlyingSynchronize() override;

 private:
    Stats statsData;
    ImGuiIO* io = nullptr;
    ImGuiStyle* style = nullptr;
    WGPUCommandEncoder encoder;
    WGPURenderPassColorAttachment colorAttachments;
    WGPURenderPassDescriptor renderPassDesc;
    WGPUQueue queue;
    WGPUTextureView framebufferTexture;
    std::vector<std::shared_ptr<SurfaceImp<DeviceType::WebGPU>>> surfaces;

    std::shared_ptr<Viewport::Adapter<DeviceType::WebGPU>> viewport;

    Result recreate();
    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
};

}  // namespace Jetstream::Render

#endif
