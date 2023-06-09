#ifndef JETSTREAM_RENDER_BASE_BUFFER_HH
#define JETSTREAM_RENDER_BASE_BUFFER_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"

namespace Jetstream::Render {

class Buffer {
 public:
    enum class Target : U64 {
        VERTEX,
        VERTEX_INDICES,
        STORAGE,
        UNIFORM,
        STORAGE_DYNAMIC,
        UNIFORM_DYNAMIC,
    };

    struct Config {
        U64 size;
        Target target;
        U64 elementByteSize;
        void* buffer = nullptr;
        // The underlying buffer memory has to be 
        // page-aligned for the zero-copy to work.
        bool enableZeroCopy = false;
    };

    explicit Buffer(const Config& config) : config(config) {
        JST_DEBUG("Buffer initialized.");
    }
    virtual ~Buffer() = default;

    constexpr U64 size() const {
        return config.size;
    }

    constexpr U64 byteSize() const {
        return config.size * config.elementByteSize;
    }

    virtual Result update() = 0;
    virtual Result update(const U64& offset, const U64& size) = 0;

    template<Device D> 
    static std::shared_ptr<Buffer> Factory(const Config& config) {
        return std::make_shared<BufferImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
