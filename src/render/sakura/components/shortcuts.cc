#include <jetstream/render/sakura/components/shortcuts.hh>

#include <jetstream/macros.hh>

#include "../context.hh"

#include <utility>

namespace Jetstream::Sakura {

namespace {

bool Matches(Shortcuts::Modifier modifier, const KeyModifiers& modifiers) {
    const bool primary = modifiers.control || modifiers.super;
    switch (modifier) {
        case Shortcuts::Modifier::Any:
            return true;
        case Shortcuts::Modifier::None:
            return !primary && !modifiers.shift && !modifiers.alt;
        case Shortcuts::Modifier::Primary:
            return primary && !modifiers.shift && !modifiers.alt;
        case Shortcuts::Modifier::PrimaryAlt:
            return primary && modifiers.alt && !modifiers.shift;
        case Shortcuts::Modifier::PrimaryShift:
            return primary && modifiers.shift && !modifiers.alt;
    }
    return false;
}

}  // namespace

std::string Shortcuts::Label(Modifier modifier, std::string_view key) {
#if defined(JST_OS_MAC)
    constexpr std::string_view primary = "CMD+";
    constexpr std::string_view alt = "OPTION+";
#else
    constexpr std::string_view primary = "CTRL+";
    constexpr std::string_view alt = "ALT+";
#endif
    std::string label;
    switch (modifier) {
        case Modifier::Any:
        case Modifier::None:
            break;
        case Modifier::Primary:
            label = primary;
            break;
        case Modifier::PrimaryAlt:
            label = std::string(primary) + std::string(alt);
            break;
        case Modifier::PrimaryShift:
            label = std::string(primary) + "SHIFT+";
            break;
    }
    return label + std::string(key);
}

const char* Shortcuts::PrimaryName() {
#if defined(JST_OS_MAC)
    return "Cmd";
#else
    return "Ctrl";
#endif
}

struct Shortcuts::Impl {
    Config config;
};

Shortcuts::Shortcuts() {
    impl = std::make_unique<Impl>();
}

Shortcuts::~Shortcuts() = default;

void Shortcuts::update(Config config) {
    const bool changed = config.content != impl->config.content || (config.content && !hasChild(*config.content));
    impl->config = std::move(config);
    if (changed) {
        mountChildren(impl->config.content ? std::vector<Component*>{impl->config.content} : std::vector<Component*>{});
    }
}

Extent2D<F32> Shortcuts::measure(const Context& ctx, Extent2D<F32> available) {
    if (!impl->config.content) {
        return {0.0f, 0.0f};
    }
    return measureChild(*impl->config.content, ctx, available);
}

void Shortcuts::layout(const Context& ctx) {
    if (impl->config.content) {
        layoutChild(ctx, *impl->config.content, frame());
    }
}

bool Shortcuts::shortcutEvent(const KeyEvent& event) {
    if (!impl->config.enabled || event.type != KeyEventType::Press) {
        return false;
    }
    for (const auto& binding : impl->config.bindings) {
        if (binding.key == event.key && (binding.repeat || !event.repeat) &&
            Matches(binding.modifier, event.modifiers) && binding.onPressed &&
            !(binding.yieldsToText && textEditorFocused())) {
            binding.onPressed();
            return true;
        }
    }
    return false;
}

}  // namespace Jetstream::Sakura
