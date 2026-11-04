#ifndef JETSTREAM_RENDER_COMPONENTS_AXIS_HH
#define JETSTREAM_RENDER_COMPONENTS_AXIS_HH

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/logger.hh"

#include "jetstream/render/base/surface.hh"
#include "jetstream/render/components/generic.hh"
#include "jetstream/render/components/font.hh"

namespace Jetstream::Render::Components {

class JETSTREAM_API Axis : public Generic {
 public:
    using TickFormatter = std::function<std::string(F32)>;

    // Round tick values (1, 2 or 5 times a power of ten) inside a data range.
    struct TickScale {
        F64 step = 0.0;
        U64 minorDivisions = 5;
        U64 decimals = 0;
        std::vector<F64> values;
    };

    // Ticks strictly inside (left, right) with the smallest round step that
    // is at least minimumStep.
    static TickScale ComputeTicks(F64 left, F64 right, F64 minimumStep);

    struct Config {
        F32 thickness = 1.0f;
        bool showInteriorGrid = true;
        F32 verticalScale = 1.0f;
        bool showFrameTicks = false;
        F32 majorTickLengthPx = 16.0f;
        F32 minorTickLengthPx = 12.0f;
        F32 minXLabelSpacingPx = 150.0f;
        F32 minYLabelSpacingPx = 150.0f;
        F32 labelCollisionPaddingPx = 4.0f;
        ColorRGBA<F32> gridColor = {0.2f, 0.2f, 0.2f, 1.0f};
        ColorRGBA<F32> majorGridColor = {0.4f, 0.4f, 0.4f, 1.0f};
        ColorRGBA<F32> labelColor = {1.0f, 1.0f, 1.0f, 1.0f};
        std::string xTitle;
        std::string yTitle;
        bool yLabelOnRight = false;
        bool yLabelOutside = false;
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
    Result updateZoom(F32 zoom, F32 translation);
    Result updateVerticalScale(F32 scale);
    Result updateScissorRect(const Render::ScissorRect& rect);

    Result setShowFrameTicks(bool visible);

    Result updateTickFormatters(TickFormatter xFormatter,
                                TickFormatter yFormatter = {});

    Result updateTitles(const std::string& xTitle,
                        const std::string& yTitle);

    // Place the vertical grid lines and X labels on round values of the data
    // range shown between the left and right frame edges. This replaces the
    // evenly divided grid and the X tick formatter until cleared.
    Result updateXRange(F64 left, F64 right);
    Result clearXRange();

    const Extent2D<F32>& paddingScale() const;
    const TickScale& currentXTicks() const;
    U64 currentVerticalLineCount() const;
    U64 currentHorizontalLineCount() const;

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
