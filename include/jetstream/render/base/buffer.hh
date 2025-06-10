#ifndef JETSTREAM_RENDER_BASE_BUFFER_HH
#define JETSTREAM_RENDER_BASE_BUFFER_HH

#include <memory>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"
#include "jetstream/render/types.hh"
#include "jetstream/render/base/implementations.hh"
#include "jetstream/render/base/window_attachment.hh"

namespace Jetstream::Render {

class Buffer : public WindowAttachment {
 public:
    enum class Target : U16 {
        VERTEX          = 1 << 0,
        VERTEX_INDICES  = 1 << 1,
        STORAGE         = 1 << 2,
        UNIFORM         = 1 << 3,
        STORAGE_DYNAMIC = 1 << 4,
        UNIFORM_DYNAMIC = 1 << 5,
        INDIRECT        = 1 << 6,
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

    explicit Buffer(const Config& config) : config(config) {}
    virtual ~Buffer() = default;

    Type type() const override {
        return Type::Buffer;
    }

    const Config& getConfig() const {
        return config;
    }

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

static constexpr Buffer::Target operator&(Buffer::Target lhs, Buffer::Target rhs) {
    return static_cast<Buffer::Target>(static_cast<U16>(lhs) & static_cast<U16>(rhs));
}

static constexpr Buffer::Target operator|(Buffer::Target lhs, Buffer::Target rhs) {
    return static_cast<Buffer::Target>(static_cast<U16>(lhs) | static_cast<U16>(rhs));
}

}  // namespace Jetstream::Render

#endif
