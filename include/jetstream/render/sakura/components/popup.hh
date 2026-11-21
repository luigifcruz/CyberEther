#ifndef JETSTREAM_RENDER_SAKURA_POPUP_HH
#define JETSTREAM_RENDER_SAKURA_POPUP_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>

namespace Jetstream::Sakura {

enum class PopupPlacement {
    Below,
    BelowEnd,
    Point,
    Side,
    Tooltip,
    Center,
    Inside,
};

struct Popup : public Component {
    struct Config {
        Component* content = nullptr;
        bool open = false;
        PopupPlacement placement = PopupPlacement::Below;
        bool atCursor = false;
        bool panel = false;
        F32 padding = 12.0f;
        F32 maxWidth = 0.0f;
        Extent2D<F32> size = {0.0f, 0.0f};
        Callback<void()> onDismiss;
        bool passthroughAnchor = false;

        bool operator==(const Config&) const = default;
    };

    Popup();
    ~Popup();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_POPUP_HH
