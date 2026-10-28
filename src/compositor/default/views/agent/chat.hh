#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH

#include "placeholder.hh"
#include "../components/assistant_composer.hh"

#include "jetstream/render/sakura/components/chat.hh"
#include "jetstream/render/sakura/components/dock.hh"
#include "jetstream/render/sakura/typography.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <functional>
#include <utility>

namespace Jetstream {

struct AgentChatView {
    static constexpr F32 Width = 440.0f;

    struct Config {
        bool open = false;
        bool hidden = true;
        U64 focusRequest = 0;
        std::function<void()> onClose;
        AgentChatPlaceholder::Config placeholder;
        Sakura::Chat::Config chat;
    };

    void update(Config config) {
        hidden = config.hidden;
        open = config.open;

        config.chat.id = "agent-chat";
        config.chat.fontSize = Sakura::Typography::FontSize;
        config.chat.maxContentWidth = kMaxContentWidth;
        config.chat.focusRequest = config.focusRequest;
        config.chat.backgroundColorKey = "transparent";
        config.chat.surfaceColorKey = "tile_bg";
        config.chat.composer = AssistantComposer::Style({
            .backgroundColorKey = "agent_composer",
            .rounding = Sakura::Rounding::Card,
            .chrome = true,
            .clearButton = false,
        });
        config.chat.voiceHint = AssistantComposer::VoiceHint;
        placeholder.update(std::move(config.placeholder));
        config.chat.placeholder = &placeholder.component();
        auto onClear = config.chat.onClear;
        chat.update(std::move(config.chat));

        dock.update({
            .accepts = "assistant",
            .panes = {{
                .id = "assistant",
                .title = "Assistant",
                .group = "assistant",
                .icon = ICON_LC_WAND_SPARKLES,
                .content = &chat,
                .maximizable = false,
                .padding = Sakura::Dock::ContentPadding,
                .onClose = std::move(config.onClose),
                .actions = {{.icon = ICON_LC_SQUARE_PEN, .onClick = std::move(onClear)}},
            }},
            .resizable = false,
        });
    }

    bool shown() const {
        return open && !hidden;
    }

    Sakura::Component& component() {
        return dock;
    }

 private:
    static constexpr F32 kMaxContentWidth = 760.0f;

    AgentChatPlaceholder placeholder;
    Sakura::Chat chat;
    Sakura::Dock dock;
    bool hidden = true;
    bool open = false;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH
