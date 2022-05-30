#ifndef JETSTREAM_RENDER_METAL_BUFFER_HH
#define JETSTREAM_RENDER_METAL_BUFFER_HH

#include "jetstream/render/base/buffer.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class BufferImp<Device::Metal> : public Buffer {
 public:
    explicit BufferImp(const Config& config);

    using Render::Buffer::size;

    const Result update();
    const Result update(const U64& offset, const U64& size);

 protected:
    const Result create();
    const Result destroy();

    constexpr const MTL::Buffer* getHandle() const {
        return buffer;
    }

 private:
    MTL::Buffer* buffer = nullptr;

    friend class SurfaceImp<Device::Metal>;
    friend class ProgramImp<Device::Metal>;
    friend class VertexImp<Device::Metal>;
};

}  // namespace Jetstream::Render

#endif
