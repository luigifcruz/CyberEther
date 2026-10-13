#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH

#include "jetstream/render/sakura/components/retained/chat_composer.hh"
#include "jetstream/render/sakura/typography.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

namespace Jetstream {

inline Sakura::Retained::ChatComposer::Style AssistantComposerStyle(Sakura::Retained::ChatComposer::Style style,
                                                                     F32 pixelRatio) {
    style.placeholder = "Ask anything...";
    style.icon = ICON_FA_WAND_MAGIC_SPARKLES;
    style.iconColorKey = "text_secondary";
    style.minLines = 1.0f;
    style.maxLines = 5.0f;
    style.fontSize = Sakura::Typography::FontSize * pixelRatio;
    style.padding *= pixelRatio;
    style.borderWidth *= pixelRatio;
    if (style.cornerRadius) {
        *style.cornerRadius *= pixelRatio;
    }
    style.margin = {
        style.margin.left * pixelRatio,
        style.margin.top * pixelRatio,
        style.margin.right * pixelRatio,
        style.margin.bottom * pixelRatio,
    };
    return style;
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_ASSISTANT_COMPOSER_HH
