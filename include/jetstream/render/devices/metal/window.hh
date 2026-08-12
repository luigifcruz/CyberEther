#ifndef JETSTREAM_RENDER_METAL_WINDOW_HH
#define JETSTREAM_RENDER_METAL_WINDOW_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/render/devices/metal/transfer.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/base.hh"

namespace Jetstream::Render {

template<>
class JETSTREAM_API WindowImp<DeviceType::Metal> : public Window {
 public:
    explicit WindowImp(const Config& config,
                       const std::shared_ptr<Viewport::Adapter<DeviceType::Metal>>& viewport);

    const Stats& stats() const override;
    std::string info() const override;

    constexpr DeviceType device() const override {
        return DeviceType::Metal;
    };

 protected:
    Result bindSurface(const std::shared_ptr<Surface>& surface) override;
    Result unbindSurface(const std::shared_ptr<Surface>& surface) override;

    Result underlyingCreate() override;
    Result underlyingDestroy() override;

    Result underlyingBegin() override;
    Result underlyingEnd() override;
    Result underlyingCancel() override;

    Result underlyingSynchronize() override;

 private:
    static constexpr size_t FramesInFlight = 3;

    Stats statsData;
    ImGuiIO* io = nullptr;
    ImGuiStyle* style = nullptr;
    MTL::Device* dev = nullptr;
    NS::AutoreleasePool* innerPool = nullptr;
    NS::AutoreleasePool* outerPool = nullptr;
    CA::MetalDrawable* drawable = nullptr;
    MTL::CommandQueue* commandQueue = nullptr;
    MTL::CommandBuffer* commandBuffer = nullptr;
    size_t currentFrame = 0;
    std::array<MTL::CommandBuffer*, FramesInFlight> inFlightCommandBuffers{};
    TransferImp<DeviceType::Metal> transferEncoder;
    MTL::RenderPassDescriptor* renderPassDescriptor = nullptr;
    bool imguiCreated = false;
    std::vector<std::shared_ptr<SurfaceImp<DeviceType::Metal>>> surfaces;
    std::shared_ptr<Viewport::Adapter<DeviceType::Metal>> viewport;

    Result createImgui();
    Result destroyImgui();
    Result beginImgui();
    Result endImgui();
};

}  // namespace Jetstream::Render

#endif
