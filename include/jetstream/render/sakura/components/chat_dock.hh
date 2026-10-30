#ifndef JETSTREAM_RENDER_SAKURA_CHAT_DOCK_HH
#define JETSTREAM_RENDER_SAKURA_CHAT_DOCK_HH

#include <jetstream/render/sakura/components/chat.hh>
#include <jetstream/render/sakura/components/layer.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct ChatDock : public Layer {
    struct Config {
        std::string id;
        std::string title;
        bool open = false;
        F32 fontSize = Typography::FontSize;
        F32 topOffset = 0.0f;
        F32 margin = 12.0f;
        Extent2D<F32> size = {440.0f, 680.0f};
        Extent2D<F32> minimumSize = {360.0f, 320.0f};
        F32 resizeGrip = 6.0f;
        F32 compactWidth = 240.0f;
        F32 rounding = 16.0f;
        F32 cardRounding = 20.0f;
        F32 headerHeight = 30.0f;
        F32 outlineWidth = 1.0f;
        std::string backgroundColorKey = "card";
        std::string outlineColorKey = "border";
        std::string alertColorKey = "accent_color";
        Chat::Config chat;
        Callback<void()> onOpen;
        Callback<void()> onMinimize;

        bool operator==(const Config&) const = default;
    };

    ChatDock();
    ~ChatDock();

    void update(Config config);
    bool covers(const Extent2D<F32>& point) const override;

 protected:
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorOverChildren(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_CHAT_DOCK_HH
