#ifndef JETSTREAM_RENDER_COMPONENTS_TEXT_HH
#define JETSTREAM_RENDER_COMPONENTS_TEXT_HH

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/components/generic.hh"
#include "jetstream/render/components/font.hh"

namespace Jetstream::Render::Components {

class Text : public Generic {
 public:
    struct Config {
        U64 maxCharacters = 32;
        std::shared_ptr<Font> font;
    };

    Text(const Config& config);
    ~Text();

    Result create(Window* window);
    Result destroy(Window* window);

    Result surface(Render::Surface::Config& config);

    void put(const F32& scale,
             const Extent2D<F32>& position, 
             const Extent2D<F32>& pixelSize, 
             const bool& center = false,
             const F32& rotationDeg = 0.0f);
    void fill(const std::string& text);

    constexpr const Config& getConfig() const {
        return config;
    }

 private:
    Config config;

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    friend class Text;
};

}  // namespace Jetstream::Render::Components

#endif