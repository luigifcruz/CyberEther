#ifndef JETSTREAM_RENDER_WEBGPU_WINDOW_HH
#define JETSTREAM_RENDER_WEBGPU_WINDOW_HH

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

    const Stats& stats() const override;
    void drawDebugMessage() const override;

    constexpr Device device() const override {
        return Device::WebGPU;
    };

 protected:
    Result bindBuffer(const std::shared_ptr<Buffer>& buffer) override;
    Result unbindBuffer(const std::shared_ptr<Buffer>& buffer) override;

    Result bindTexture(const std::shared_ptr<Texture>& texture) override;
    Result unbindTexture(const std::shared_ptr<Texture>& texture) override;

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
    wgpu::CommandEncoder encoder;

    wgpu::RenderPassColorAttachment colorAttachments;
    wgpu::RenderPassDescriptor renderPassDesc;
    wgpu::Queue queue;

    template<typename T>
    Result bindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container);

    template<typename T>
    Result unbindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container);

    std::vector<std::shared_ptr<BufferImp<Device::WebGPU>>> buffers;
    std::vector<std::shared_ptr<TextureImp<Device::WebGPU>>> textures;
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
