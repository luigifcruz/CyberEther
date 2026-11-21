#ifndef JETSTREAM_RENDER_SAKURA_SURFACE_IMAGE_HH
#define JETSTREAM_RENDER_SAKURA_SURFACE_IMAGE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/surface.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>

namespace Jetstream::Render {
class Texture;
}  // namespace Jetstream::Render

namespace Jetstream::Sakura {

struct SurfaceImage : public Component {
    struct Config {
        std::shared_ptr<const Render::Texture> texture;
        F32 rounding = 0.0f;
        Callback<void(const SurfaceResize&)> onSize;
        Callback<void(InputEvent)> onInput;
        Callback<SurfaceCursor()> resolveCursor;

        bool operator==(const Config&) const = default;
    };

    SurfaceImage();
    ~SurfaceImage();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    bool keyEvent(const KeyEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;
    void focusChanged(bool focused) override;
    void wentOnscreen(U64 frame) override;
    void wentOffscreen(U64 frame) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SURFACE_IMAGE_HH
