#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH

#include "../components/assistant_composer.hh"

#include "jetstream/render/sakura/components/retained/chat_dock.hh"
#include "jetstream/render/sakura/components/retained/canvas.hh"
#include "jetstream/render/sakura/components/layer.hh"
#include "jetstream/render/sakura/typography.hh"

namespace Jetstream {

struct AgentChatView {
    struct Config {
        bool open = false;
        bool hidden = false;
        U64 focusRequest = 0;
        F32 topOffset = 0.0f;
        std::function<void()> onOpen;
        std::function<void()> onMinimize;
        Sakura::Retained::Chat::Config chat;
    };

    AgentChatView() {
        canvas.mount(dock);
    }

    void update(Config config) {
        hidden = config.hidden;
        const F32 r = pixelRatio;

        config.chat.id = "agent-chat";
        config.chat.fontSize = Sakura::Typography::FontSize * r;
        config.chat.maxContentWidth = kMaxContentWidth * r;
        config.chat.focusRequest = config.focusRequest;
        config.chat.backgroundColorKey = "transparent";
        config.chat.surfaceColorKey = "agent_panel";
        const Sakura::Retained::Chat::ComposerStyle composer{
            .backgroundColorKey = "agent_composer",
            .clearButton = false,
        };
        config.chat.composer = AssistantComposerStyle(composer, r);

        dock.update({
            .id = "agent-dock",
            .title = "Assistant",
            .open = config.open,
            .fontSize = Sakura::Typography::FontSize * r,
            .margin = kMargin * r,
            .size = {kCardWidth * r, kCardHeight * r},
            .minSize = {kMinCardWidth * r, kMinCardHeight * r},
            .resizeGrip = kResizeGrip * r,
            .compactWidth = kCompactWidth * r,
            .cornerRadius = kComposerRadius * r,
            .cardRadius = kCardRadius * r,
            .headerHeight = kHeaderHeight * r,
            .outlineWidth = kOutlineWidth * r,
            .backgroundColorKey = "agent_panel",
            .outlineColorKey = "agent_outline",
            .alertColorKey = "agent_activity",
            .chat = std::move(config.chat),
            .onOpen = std::move(config.onOpen),
            .onMinimize = [this, onMinimize = std::move(config.onMinimize)]() {
                ++blurRequests;
                if (onMinimize) {
                    onMinimize();
                }
            },
        });

        canvas.update({
            .id = "agent-dock:canvas",
            .passthrough = true,
            .clearColor = {0.0f, 0.0f, 0.0f, 0.0f},
            .onLayout = [this](const Sakura::Retained::Canvas::Layout& layout) {
                pixelRatio = layout.pixelRatio;
            },
        });

        layer.update({
            .id = "agent-dock:layer",
            .anchor = Sakura::Layer::Anchor::BottomRight,
            .topOffset = config.topOffset,
            .focusRequest = config.focusRequest,
            .blurRequest = blurRequests,
        });
    }

    void render(const Sakura::Context& context) {
        if (hidden) {
            return;
        }

        layer.render(context, [this](const Sakura::Context& ctx) {
            canvas.render(ctx);
        });
    }

 private:
    static constexpr F32 kMargin = 12.0f;
    static constexpr F32 kMaxContentWidth = 760.0f;
    static constexpr F32 kCardWidth = 440.0f;
    static constexpr F32 kCardHeight = 680.0f;
    static constexpr F32 kMinCardWidth = 360.0f;
    static constexpr F32 kMinCardHeight = 320.0f;
    static constexpr F32 kResizeGrip = 6.0f;
    static constexpr F32 kCompactWidth = 240.0f;
    static constexpr F32 kComposerRadius = 16.0f;
    static constexpr F32 kCardRadius = 20.0f;
    static constexpr F32 kHeaderHeight = 30.0f;
    static constexpr F32 kOutlineWidth = 1.5f;

    Sakura::Retained::ChatDock dock;
    Sakura::Retained::Canvas canvas;
    Sakura::Layer layer;
    U64 blurRequests = 0;
    F32 pixelRatio = 1.0f;
    bool hidden = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH
