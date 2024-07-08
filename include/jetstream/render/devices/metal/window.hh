#ifndef JETSTREAM_RENDER_METAL_WINDOW_HH
#define JETSTREAM_RENDER_METAL_WINDOW_HH

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

    const Stats& stats() const override;
    void drawDebugMessage() const override;

    constexpr Device device() const override {
        return Device::Metal;
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
    MTL::Device* dev = nullptr;
    NS::AutoreleasePool* innerPool;
    NS::AutoreleasePool* outerPool;
    CA::MetalDrawable* drawable = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    MTL::CommandBuffer* commandBuffer = nullptr;
    MTL::RenderPassDescriptor* renderPassDescriptor = nullptr;

    template<typename T>
    Result bindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container);

    template<typename T>
    Result unbindResource(const auto& resource, std::vector<std::shared_ptr<T>>& container);

    std::vector<std::shared_ptr<BufferImp<Device::Metal>>> buffers;
    std::vector<std::shared_ptr<TextureImp<Device::Metal>>> textures;
    std::vector<std::shared_ptr<SurfaceImp<Device::Metal>>> surfaces;

    std::shared_ptr<Viewport::Adapter<Device::Metal>> viewport;

    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
};

}  // namespace Jetstream::Render

#endif
