#ifndef JETSTREAM_RENDER_BASE_WINDOW_ATTACHMENT_HH
#define JETSTREAM_RENDER_BASE_WINDOW_ATTACHMENT_HH\

#include "jetstream/backend/base.hh"

namespace Jetstream::Render {

class WindowAttachment {
 public:
    WindowAttachment() = default;
    virtual ~WindowAttachment() = default;

    enum class Type : uint8_t {
        Unknown = 0,
        Surface,
        Texture,
        Buffer
    };

    virtual Type type() const = 0;

    virtual Result create() = 0;
    virtual Result destroy() = 0;
};

}  // namespace Jetstream::Render

#endif
