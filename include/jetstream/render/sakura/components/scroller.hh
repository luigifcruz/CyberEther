#ifndef JETSTREAM_RENDER_SAKURA_SCROLLER_HH
#define JETSTREAM_RENDER_SAKURA_SCROLLER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct Scroller : public Component {
    struct Config {
        std::string id;
        F32 contentWidth = 0.0f;
        F32 contentHeight = 0.0f;
        F32 scrollX = 0.0f;
        F32 scrollY = 0.0f;
        bool scrollbar = true;
        bool wheel = true;
        F32 wheelStep = 48.0f;
        F32 thickness = 8.0f;
        F32 margin = 3.0f;
        Extent2D<F32> trackInset = {0.0f, 0.0f};
        F32 thumbInset = 0.0f;
        std::string trackColorKey = "editor_scrollbar_track";
        std::string thumbColorKey = "editor_scrollbar_thumb";
        Callback<void(Rect contentRect, std::optional<Rect> clipRect)> onLayout;
        Callback<void(F32 scrollY)> onScrollY;
        Callback<void(F32 scrollX)> onScrollX;

        bool operator==(const Config&) const = default;
    };

    Scroller();
    ~Scroller();

    void update(Config config);

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SCROLLER_HH
