#ifndef JETSTREAM_RENDER_COMPONENTS_TEXT_HH
#define JETSTREAM_RENDER_COMPONENTS_TEXT_HH

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/components/generic.hh"
#include "jetstream/render/components/font.hh"

namespace Jetstream::Render::Components {

class Text : public Generic {
 public:
    struct ElementConfig {
        F32 scale = 1.0f;
        Extent2D<F32> position = {0.0f, 0.0f};
        Extent2D<I32> alignment = {0, 0};
        F32 rotationDeg = 0.0f;
        std::string fill = "";
    };

    struct Config {
        U64 maxCharacters = 32;
        std::shared_ptr<Font> font;
        Extent2D<F32> pixelSize = {0.0f, 0.0f};
        ColorRGBA<F32> color = {1.0f, 1.0f, 1.0f, 1.0f};
        std::unordered_map<std::string, ElementConfig> elements;
        F32 sharpness = 0.5f;
    };

    Text(const Config& config);
    ~Text();

    Result create(Window* window);
    Result destroy(Window* window);

    Result surface(Render::Surface::Config& config);

    Result present();

    const ElementConfig& get(const std::string& elementId) const;
    Result update(const std::string& elementId, const ElementConfig& elementConfig);

    Result updatePixelSize(const Extent2D<F32>& pixelSize);

    constexpr const Config& getConfig() const {
        return config;
    }

 private:
    Config config;

    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace Jetstream::Render::Components

#endif
