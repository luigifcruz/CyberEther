#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_METRICS_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_METRICS_HH

#include <jetstream/render/base/window.hh>
#include <jetstream/render/components/text.hh>
#include <jetstream/types.hh>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Jetstream::Sakura::Retained {

struct TextMetrics {
    Render::Window* window = nullptr;
    std::unordered_map<std::string, std::unique_ptr<Render::Components::Text>> labels;
    std::unordered_map<std::string, F32> widths;

    void setWindow(Render::Window* nextWindow) {
        if (window == nextWindow) {
            return;
        }
        window = nextWindow;
        labels.clear();
        widths.clear();
    }

    Render::Components::Text* label(const std::string& fontName) {
        if (!window) {
            return nullptr;
        }

        std::string resolved = fontName.empty() ? std::string("default_mono") : fontName;
        if (!window->hasFont(resolved)) {
            if (!window->hasFont("default_mono")) {
                return nullptr;
            }
            resolved = "default_mono";
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

    F32 measure(const std::string& fontName, const std::string& value, F32 fontSize) {
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

    std::vector<F32> advances(const std::string& fontName, const std::string& value, F32 fontSize) {
        auto* text = label(fontName);
        if (!text) {
            return std::vector<F32>(value.size(), 0.0f);
        }
        return text->advances(value, scale(*text, fontSize));
    }

 private:
    static constexpr U64 kWidthCacheCapacity = 256;

    static F32 scale(const Render::Components::Text& text, F32 fontSize) {
        if (!text.getConfig().font) {
            return 0.0f;
        }
        return fontSize / std::max(1e-3f, text.getConfig().font->getConfig().size);
    }
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_METRICS_HH
