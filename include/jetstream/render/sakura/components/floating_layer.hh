#ifndef JETSTREAM_RENDER_SAKURA_FLOATING_LAYER_HH
#define JETSTREAM_RENDER_SAKURA_FLOATING_LAYER_HH

#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct FloatingLayer : public Layer {
    static constexpr F32 Rounding = 12.0f;

    struct Window {
        std::string id;
        std::string title;
        Component* content = nullptr;
        Extent2D<F32> size = {500.0f, 400.0f};
        F32 padding = 0.0f;
        Callback<void()> onClose;
        Callback<void()> onFocus;

        bool operator==(const Window&) const = default;
    };

    struct Config {
        std::vector<Window> windows;

        bool operator==(const Config&) const = default;
    };

    FloatingLayer();
    ~FloatingLayer();

    void update(Config config);
    bool covers(const Extent2D<F32>& point) const override;

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    bool deliverTo(const std::string& id, const MouseEvent& event);
    void pointTo(const std::string& id, const MouseEvent& event);
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_FLOATING_LAYER_HH
