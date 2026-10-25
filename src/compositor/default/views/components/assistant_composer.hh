#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH

#include "jetstream/render/sakura/components/chat_composer.hh"
#include "jetstream/render/sakura/typography.hh"
#include "jetstream/render/tools/icons_lucide.hh"

namespace Jetstream {

inline Sakura::ChatComposer::Style AssistantComposerStyle(Sakura::ChatComposer::Style style) {
    style.placeholder = "Ask anything...";
    style.icon = ICON_LC_WAND_SPARKLES;
    style.iconColorKey = "text_secondary";
    style.minLines = 1.0f;
    style.maxLines = 5.0f;
    style.fontSize = Sakura::Typography::FontSize;
    return style;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH
