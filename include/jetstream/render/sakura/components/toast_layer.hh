#ifndef JETSTREAM_RENDER_SAKURA_TOAST_LAYER_HH
#define JETSTREAM_RENDER_SAKURA_TOAST_LAYER_HH

#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>

namespace Jetstream::Sakura {

struct ToastLayer : public Layer {
    ToastLayer();
    ~ToastLayer();

    void update();
    bool covers(const Extent2D<F32>& point) const override;

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TOAST_LAYER_HH
