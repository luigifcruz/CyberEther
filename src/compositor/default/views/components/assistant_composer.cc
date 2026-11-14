#include "assistant_composer.hh"

namespace Jetstream {

const char* AssistantComposer::RandomPrompt() {
    std::random_device device;
    return Prompts[std::uniform_int_distribution<U64>(0, Prompts.size() - 1)(device)];
}

Sakura::ChatComposer::Style AssistantComposer::Style(Sakura::ChatComposer::Style style,
                                                     const char* placeholder) {
    style.placeholder = placeholder;
    style.iconColorKey = "text_secondary";
    style.minLines = 1.0f;
    style.maxLines = 5.0f;
    style.fontSize = Sakura::Typography::FontSize;
    return style;
}

}  // namespace Jetstream
