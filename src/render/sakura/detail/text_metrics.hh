#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_TEXT_METRICS_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_TEXT_METRICS_HH

#include <jetstream/render/base/window.hh>
#include <jetstream/render/components/text.hh>
#include <jetstream/types.hh>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Jetstream::Sakura {

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

    Render::Components::Text* label(const std::string& fontName);

    F32 measure(const std::string& fontName, const std::string& value, F32 fontSize);

    std::vector<F32> advances(const std::string& fontName, const std::string& value, F32 fontSize) {
        auto* text = label(fontName);
        if (!text) {
            return std::vector<F32>(value.size(), 0.0f);
        }
        return text->advances(value, scale(*text, fontSize));
    }

    std::string ellipsize(const std::string& fontName, const std::string& text, F32 fontSize, F32 width);

 private:
    static constexpr U64 kWidthCacheCapacity = 256;

    static F32 scale(const Render::Components::Text& text, F32 fontSize) {
        if (!text.getConfig().font) {
            return 0.0f;
        }
        return fontSize / std::max(1e-3f, text.getConfig().font->getConfig().size);
    }
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_TEXT_METRICS_HH
