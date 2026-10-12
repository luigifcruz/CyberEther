#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_DOCK_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_DOCK_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/render/sakura/components/retained/chat.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura::Retained {

struct ChatDock : public Component {
    struct Config {
        std::string id;
        std::string title;
        bool open = false;
        F32 fontSize = Typography::FontSize;
        F32 margin = 12.0f;
        Extent2D<F32> size = {440.0f, 680.0f};
        Extent2D<F32> minSize = {360.0f, 320.0f};
        F32 resizeGrip = 6.0f;
        F32 compactWidth = 240.0f;
        F32 cornerRadius = 16.0f;
        F32 cardRadius = 20.0f;
        F32 headerHeight = 30.0f;
        F32 outlineWidth = 1.0f;
        std::string backgroundColorKey = "card";
        std::string outlineColorKey = "border";
        std::string alertColorKey = "accent_color";
        Chat::Config chat;
        std::function<void()> onOpen;
        std::function<void()> onMinimize;
    };

    ChatDock();
    ~ChatDock();

    ChatDock(const ChatDock&) = delete;
    ChatDock& operator=(const ChatDock&) = delete;

    bool update(Config config);

 protected:
    void layout(const Context& ctx) override;
    bool event(const MouseEvent& event) override;
    bool hitTest(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_CHAT_DOCK_HH
