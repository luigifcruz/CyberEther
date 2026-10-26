#include "text_metrics.hh"

#include <jetstream/render/sakura/typography.hh>

namespace Jetstream::Sakura {

Render::Components::Text* TextMetrics::label(const std::string& fontName) {
    if (!window) {
        return nullptr;
    }

    std::string resolved = fontName.empty() ? std::string(Typography::MonoFont) : fontName;
    if (!window->hasFont(resolved)) {
        if (!window->hasFont(Typography::MonoFont)) {
            return nullptr;
        }
        resolved = Typography::MonoFont;
    }

    auto it = labels.find(resolved);
    if (it != labels.end()) {
        return it->second.get();
    }

    Render::Components::Text::Config config;
    config.maxCharacters = 1;
    config.font = window->font(resolved);
    auto inserted = labels.emplace(resolved, std::make_unique<Render::Components::Text>(config)).first;
    return inserted->second.get();
}

F32 TextMetrics::measure(const std::string& fontName, const std::string& value, F32 fontSize) {
    auto* text = label(fontName);
    if (!text) {
        return 0.0f;
    }
    const auto nameSize = fontName.size();
    const auto* font = text->getConfig().font.get();
    std::string key(reinterpret_cast<const char*>(&nameSize), sizeof(nameSize));
    key += fontName;
    key.append(reinterpret_cast<const char*>(&fontSize), sizeof(fontSize));
    key.append(reinterpret_cast<const char*>(&font), sizeof(font));
    key += value;
    if (const auto it = widths.find(key); it != widths.end()) {
        return it->second;
    }
    const F32 width = text->advance(value, scale(*text, fontSize));
    if (widths.size() >= kWidthCacheCapacity) {
        widths.clear();
    }
    widths.emplace(std::move(key), width);
    return width;
}

std::string TextMetrics::ellipsize(const std::string& fontName, const std::string& text, F32 fontSize, F32 width) {
    if (text.empty() || measure(fontName, text, fontSize) <= width) {
        return text;
    }
    const F32 budget = std::max(0.0f, width - measure(fontName, "...", fontSize));
    const auto steps = advances(fontName, text, fontSize);
    F32 used = 0.0f;
    U64 cut = 0;
    while (cut < text.size() && cut < steps.size() && used + steps[cut] <= budget) {
        used += steps[cut];
        ++cut;
    }
    while (cut > 0 && cut < text.size() && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) {
        --cut;
    }
    return text.substr(0, cut) + "...";
}

}  // namespace Jetstream::Sakura
