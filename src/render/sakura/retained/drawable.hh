#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_DRAWABLE_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_DRAWABLE_HH

#include "../context.hh"

namespace Jetstream::Sakura::Retained {

struct Drawable {
    virtual ~Drawable() = default;

    virtual Result attach(Context* context, Render::Surface::Config& surfaceConfig) = 0;
    virtual Result detach(Render::Window* window) = 0;
    virtual Result upload() = 0;
    virtual Result present() = 0;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_DRAWABLE_HH
