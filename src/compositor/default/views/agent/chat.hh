#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_CHAT_HH

#include "placeholder.hh"
#include "../components/assistant_composer.hh"

#include "jetstream/render/sakura/components/chat.hh"
#include "jetstream/render/sakura/components/dock.hh"
#include "jetstream/render/sakura/typography.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <utility>

namespace Jetstream {

struct AgentChatView {
    static constexpr F32 Width = 440.0f;

    struct Config {
        bool open = false;
        bool hidden = true;
        U64 focusRequest = 0;
        Sakura::Callback<void()> onClose;
        AgentChatPlaceholder::Config placeholder;
        Sakura::Chat::Config chat;
    };

    void update(Config config);
    bool shown() const;
    Sakura::Component& component();

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
