#ifndef JETSTREAM_RENDER_SAKURA_MODAL_LAYER_HH
#define JETSTREAM_RENDER_SAKURA_MODAL_LAYER_HH

#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>

namespace Jetstream::Sakura {

struct ModalLayer : public Layer {
    struct Config {
        Component* content = nullptr;
        Extent2D<F32> size = {550.0f, 0.0f};
        bool closeButton = true;
        bool chrome = true;
        bool dismissOutside = false;
        std::optional<Rect> anchor;
        Callback<void()> onClose;

        bool operator==(const Config&) const = default;
    };

    ModalLayer();
    ~ModalLayer();

    void update(Config config);
    bool covers(const Extent2D<F32>& point) const override;
    bool modal() const override;

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;

    bool shortcutEvent(const KeyEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_MODAL_LAYER_HH
