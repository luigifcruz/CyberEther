#ifndef JETSTREAM_RENDER_WEBGPU_WINDOW_HH
#define JETSTREAM_RENDER_WEBGPU_WINDOW_HH

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/base/window.hh"
#include "jetstream/render/devices/webgpu/transfer.hh"
#include "jetstream/backend/base.hh"
#include "jetstream/viewport/base.hh"

#include <webgpu/webgpu.h>

#include <memory>

namespace Jetstream::Render {

template<>
class JETSTREAM_API WindowImp<DeviceType::WebGPU> : public Window {
 public:
    explicit WindowImp(const Config& config,
                       const std::shared_ptr<Viewport::Adapter<DeviceType::WebGPU>>& viewport);

    ~WindowImp() override;

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
    Result underlyingCancel() override;

    Result underlyingSynchronize() override;
    Result underlyingRecreate() override;

    Viewport::Generic* underlyingViewport() const override { return viewport.get(); }

 private:
    struct Presenter;

    WGPUCommandEncoder encoder = nullptr;
    WGPUQueue queue = nullptr;
    WGPUTextureView framebufferTexture = nullptr;
    TransferImp<DeviceType::WebGPU> transferEncoder;
    std::vector<std::shared_ptr<SurfaceImp<DeviceType::WebGPU>>> surfaces;
    std::unique_ptr<Presenter> presenter;

    std::shared_ptr<Viewport::Adapter<DeviceType::WebGPU>> viewport;

    Result createPresenter();
    Result encodePresent();
};

}  // namespace Jetstream::Render

#endif
