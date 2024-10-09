#ifndef JETSTREAM_RENDER_COMPONENTS_FONT_HH
#define JETSTREAM_RENDER_COMPONENTS_FONT_HH

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/components/generic.hh"
#include "jetstream/render/base/texture.hh"

namespace Jetstream::Render::Components {

class Font : public Generic {
 public:
    struct Config {
        F32 size = 13.0f;
        const void* data = nullptr;
    };

    Font(const Config& config);
    ~Font();

    Result create(Window* window);
    Result destroy(Window* window);

    constexpr const Config& getConfig() const {
        return config;
    }

 protected:
    struct Glyph {
        I32 x0;
        I32 y0;
        I32 x1;
        I32 y1;
        F32 xOffset;
        F32 yOffset;
        F32 xAdvance;
    };

    const Glyph& glyph(const I32& code) const;
    const Extent2D<I32>& atlasSize() const;
    const std::shared_ptr<Render::Texture>& atlas() const;

 private:
    Config config;

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    std::unordered_map<I32, Glyph> glyphs;

    friend class Text;
};

}  // namespace Jetstream::Render::Components

#endif
