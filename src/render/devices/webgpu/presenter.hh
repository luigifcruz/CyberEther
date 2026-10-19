#ifndef JETSTREAM_RENDER_WEBGPU_PRESENTER_HH
#define JETSTREAM_RENDER_WEBGPU_PRESENTER_HH

#include "jetstream/render/base/present.hh"
#include "jetstream/render/devices/webgpu/window.hh"

#include "present_bindings.hh"

#include <optional>

namespace Jetstream::Render {

struct WindowImp<DeviceType::WebGPU>::Presenter {
    Result create(WGPUTextureFormat format);
    void destroy();
    Result encode(WGPURenderPassEncoder pass,
                  WGPUQueue queue,
                  const PresentList& list,
                  const WindowMetrics& metrics,
                  const Extent2D<U64>& target);

 private:
    WGPUDevice device = nullptr;
    WGPUBindGroupLayout bindGroupLayout = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    WGPURenderPipeline straight = nullptr;
    WGPURenderPipeline premultiplied = nullptr;
    WGPUSampler sampler = nullptr;
    WGPUBuffer vertexBuffer = nullptr;
    U64 vertexCapacity = 0;
    std::optional<PresentBindingCache<WGPUBindGroup>> bindings;

    Result reserve(U64 vertices);
};

}  // namespace Jetstream::Render

#endif  // JETSTREAM_RENDER_WEBGPU_PRESENTER_HH
