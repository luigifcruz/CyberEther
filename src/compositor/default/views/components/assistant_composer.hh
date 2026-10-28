#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH

#include "jetstream/config.hh"
#include "jetstream/render/sakura/components/chat_composer.hh"
#include "jetstream/render/sakura/typography.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/surface.hh"

namespace Jetstream {

struct AssistantComposer {
    static constexpr KeyCode VoiceKey = KeyCode::J;
#if defined(JST_OS_MAC)
    static constexpr const char* VoiceHint = "Start voice (Cmd+J)";
#else
    static constexpr const char* VoiceHint = "Start voice (Ctrl+J)";
#endif

    static Sakura::ChatComposer::Style Style(Sakura::ChatComposer::Style style) {
        style.placeholder = "Ask anything...";
        style.iconColorKey = "text_secondary";
        style.minLines = 1.0f;
        style.maxLines = 5.0f;
        style.fontSize = Sakura::Typography::FontSize;
        return style;
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH
