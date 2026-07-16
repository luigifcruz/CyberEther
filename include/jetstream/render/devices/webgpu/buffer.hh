#ifndef JETSTREAM_RENDER_WEBGPU_BUFFER_HH
#define JETSTREAM_RENDER_WEBGPU_BUFFER_HH

#include "jetstream/render/base/buffer.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class JETSTREAM_API BufferImp<DeviceType::WebGPU> : public Buffer {
 public:
    explicit BufferImp(const Config& config);

    Result create() override;
    Result destroy() override;

    using Render::Buffer::size;
    using Render::Buffer::byteSize;

 protected:
    constexpr WGPUBuffer getHandle() const {
        return buffer;
    }

 private:
    WGPUBuffer buffer;

    friend class SurfaceImp<DeviceType::WebGPU>;
    friend class ProgramImp<DeviceType::WebGPU>;
    friend class KernelImp<DeviceType::WebGPU>;
    friend class VertexImp<DeviceType::WebGPU>;
    friend class TextureImp<DeviceType::WebGPU>;
    friend class DrawImp<DeviceType::WebGPU>;
    friend class TransferImp<DeviceType::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
