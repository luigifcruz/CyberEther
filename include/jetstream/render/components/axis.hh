#ifndef JETSTREAM_RENDER_COMPONENTS_AXIS_HH
#define JETSTREAM_RENDER_COMPONENTS_AXIS_HH

#include <memory>
#include <vector>
#include <string>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/components/generic.hh"
#include "jetstream/render/components/font.hh"

namespace Jetstream::Render::Components {

class Axis : public Generic {
 public:
    struct Config {
        U64 numberOfVerticalLines = 11;
        U64 numberOfHorizontalLines = 5;
        F32 thickness = 1.0f;
        ColorRGBA<F32> gridColor = {0.2f, 0.2f, 0.2f, 1.0f};
        ColorRGBA<F32> labelColor = {1.0f, 1.0f, 1.0f, 1.0f};
        std::string xTitle;
        std::string yTitle;
        std::shared_ptr<Font> font;
        Extent2D<F32> pixelSize = {0.0f, 0.0f};
    };

    Axis(const Config& config);
    ~Axis();

    Result create(Window* window);
    Result destroy(Window* window);

    Result surfaceUnderlay(Render::Surface::Config& config);
    Result surfaceOverlay(Render::Surface::Config& config);

    Result present();

    Result updatePixelSize(const Extent2D<F32>& pixelSize);

    Result updateTickLabels(const std::vector<std::string>& xLabels,
                            const std::vector<std::string>& yLabels);

    Result updateTitles(const std::string& xTitle,
                        const std::string& yTitle);

    const Extent2D<F32>& paddingScale() const;

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
