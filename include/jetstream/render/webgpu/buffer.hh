#ifndef JETSTREAM_RENDER_WEBGPU_BUFFER_HH
#define JETSTREAM_RENDER_WEBGPU_BUFFER_HH

#include "jetstream/render/base/buffer.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class BufferImp<Device::WebGPU> : public Buffer {
 public:
    explicit BufferImp(const Config& config);

    Result create();
    Result destroy();

    using Render::Buffer::size;
    using Render::Buffer::byteSize;

    Result update();
    Result update(const U64& offset, const U64& size);

 protected:
    constexpr wgpu::Buffer& getHandle() {
        return buffer;
    }

    constexpr const wgpu::BufferBindingLayout& getBufferBindingLayout() const {
        return bufferBindingLayout;
    }

 private:
    wgpu::Buffer buffer;

    wgpu::BufferBindingLayout bufferBindingLayout;

    friend class SurfaceImp<Device::WebGPU>;
    friend class ProgramImp<Device::WebGPU>;
    friend class VertexImp<Device::WebGPU>;
};

}  // namespace Jetstream::Render

#endif
