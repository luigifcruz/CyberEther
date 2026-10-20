#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH

#include "../components/assistant_composer.hh"

#include "jetstream/render/sakura/components/chat.hh"
#include "jetstream/render/sakura/components/chat_dock.hh"
#include "jetstream/render/sakura/typography.hh"

#include <functional>
#include <utility>

namespace Jetstream {

struct AgentChatView {
    struct Config {
        bool open = false;
        bool hidden = true;
        U64 focusRequest = 0;
        F32 topOffset = 0.0f;
        std::function<void()> onOpen;
        std::function<void()> onMinimize;
        Sakura::Chat::Config chat;
    };

    void update(Config config) {
        hidden = config.hidden;

        config.chat.id = "agent-chat";
        config.chat.fontSize = Sakura::Typography::FontSize;
        config.chat.maxContentWidth = kMaxContentWidth;
        config.chat.focusRequest = config.focusRequest;
        config.chat.backgroundColorKey = "transparent";
        config.chat.surfaceColorKey = "agent_panel";
        config.chat.composer = AssistantComposerStyle({
            .backgroundColorKey = "agent_composer",
            .clearButton = false,
        });

        dock.update({
            .id = "agent-dock",
            .title = "Assistant",
            .open = config.open,
            .fontSize = Sakura::Typography::FontSize,
            .topOffset = config.topOffset,
            .margin = kMargin,
            .size = {kCardWidth, kCardHeight},
            .minimumSize = {kMinimumCardWidth, kMinimumCardHeight},
            .resizeGrip = kResizeGrip,
            .compactWidth = kCompactWidth,
            .rounding = kComposerRounding,
            .cardRounding = kCardRounding,
            .headerHeight = kHeaderHeight,
            .outlineWidth = kOutlineWidth,
            .backgroundColorKey = "agent_panel",
            .outlineColorKey = "agent_outline",
            .alertColorKey = "agent_activity",
            .chat = std::move(config.chat),
            .onOpen = std::move(config.onOpen),
            .onMinimize = std::move(config.onMinimize),
        });
    }

    bool shown() const {
        return !hidden;
    }

    Sakura::Layer& layer() {
        return dock;
    }

 private:
    static constexpr F32 kMargin = 12.0f;
    static constexpr F32 kMaxContentWidth = 760.0f;
    static constexpr F32 kCardWidth = 440.0f;
    static constexpr F32 kCardHeight = 680.0f;
    static constexpr F32 kMinimumCardWidth = 360.0f;
    static constexpr F32 kMinimumCardHeight = 320.0f;
    static constexpr F32 kResizeGrip = 6.0f;
    static constexpr F32 kCompactWidth = 240.0f;
    static constexpr F32 kComposerRounding = 16.0f;
    static constexpr F32 kCardRounding = 20.0f;
    static constexpr F32 kHeaderHeight = 30.0f;
    static constexpr F32 kOutlineWidth = 1.5f;

    Sakura::ChatDock dock;
    bool hidden = true;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH
