#include "chat.hh"

namespace Jetstream {

void AgentChatView::update(Config config) {
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

bool AgentChatView::shown() const {
    return open && !hidden;
}

Sakura::Component& AgentChatView::component() {
    return dock;
}

}  // namespace Jetstream
