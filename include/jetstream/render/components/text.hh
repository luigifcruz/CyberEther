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
        F32 scale = 1.0f;
        Extent2D<F32> position = {0.0f, 0.0f};
        Extent2D<F32> pixelSize = {0.0f, 0.0f};
        Extent2D<bool> center = {false, false};
        ColorRGBA<F32> color = {1.0f, 1.0f, 1.0f, 1.0f};
        F32 rotationDeg = 0.0f;
        std::string fill = "";
        U64 maxCharacters = 32;
        std::shared_ptr<Font> font;
    };

    Text(const Config& config);
    ~Text();

    Result create(Window* window);
    Result destroy(Window* window);

    Result surface(Render::Surface::Config& config);

    Result present();

    constexpr const F32& scale() const {
        return config.scale;
    }
    const F32& scale(const F32& scale);

    constexpr const Extent2D<F32>& position() const {
        return config.position;
    }
    const Extent2D<F32>& position(const Extent2D<F32>& position);

    constexpr const Extent2D<F32>& pixelSize() const {
        return config.pixelSize;
    }
    const Extent2D<F32>& pixelSize(const Extent2D<F32>& pixelSize);

    constexpr const Extent2D<bool>& center() const {
        return config.center;
    }
    const Extent2D<bool>& center(const Extent2D<bool>& center);

    constexpr const ColorRGBA<F32>& color() const {
        return config.color;
    }
    const ColorRGBA<F32>& color(const ColorRGBA<F32>& color);

    constexpr const F32& rotationDeg() const {
        return config.rotationDeg;
    }
    const F32& rotationDeg(const F32& rotationDeg);

    constexpr const std::string& fill() const {
        return config.fill;
    }
    const std::string& fill(const std::string& text);

    Result apply();

    constexpr const Config& getConfig() const {
        return config;
    }

 private:
    Config config;

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    Result updateVertices();
    Result updateTransform();

    friend class Text;
};

}  // namespace Jetstream::Render::Components

#endif
