#ifndef JETSTREAM_RENDER_BUFFER_HH
#define JETSTREAM_RENDER_BUFFER_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/types.hh"

namespace Jetstream::Render {

template<Device D> class BufferImp;

class Buffer {
 public:
    enum class Target : U64 {
        VERTEX,
        VERTEX_INDICES,
        STORAGE,
    };

    struct Config {
        U64 size;
        Target target;
        U64 elementByteSize;
        void* buffer = nullptr;
    };

    explicit Buffer(const Config& config) : config(config) {
        JST_DEBUG("Buffer initialized.");
    }
    virtual ~Buffer() = default;

    constexpr const U64 size() const {
        return config.size;
    }

    virtual const Result update() = 0;
    virtual const Result update(const U64& offset, const U64& size) = 0;

    template<Device D> 
    static std::shared_ptr<Buffer> Factory(const Config& config) {
        return std::make_shared<BufferImp<D>>(config);
    }

 protected:
    Config config;
};

}  // namespace Jetstream::Render

#endif
