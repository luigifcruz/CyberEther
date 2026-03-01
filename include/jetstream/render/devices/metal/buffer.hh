#ifndef JETSTREAM_RENDER_METAL_BUFFER_HH
#define JETSTREAM_RENDER_METAL_BUFFER_HH

#include "jetstream/render/base/buffer.hh"
#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

template<>
class BufferImp<DeviceType::Metal> : public Buffer {
 public:
    explicit BufferImp(const Config& config);

    Result create() override;
    Result destroy() override;

    using Render::Buffer::size;

    Result update() override;
    Result update(const U64& offset, const U64& size) override;

 protected:
    constexpr MTL::Buffer* getHandle() const {
        return buffer;
    }

 private:
    MTL::Buffer* buffer = nullptr;

    friend class SurfaceImp<DeviceType::Metal>;
    friend class ProgramImp<DeviceType::Metal>;
    friend class VertexImp<DeviceType::Metal>;
    friend class KernelImp<DeviceType::Metal>;
    friend class TextureImp<DeviceType::Metal>;
    friend class DrawImp<DeviceType::Metal>;
};

}  // namespace Jetstream::Render

#endif
