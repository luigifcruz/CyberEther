#include <jetstream/render/sakura/color.hh>

#include "base.hh"

namespace Jetstream::Sakura {

const Palette& EmptyPalette() {
    static const Palette palette;
    return palette;
}

Color ResolveColor(const Context& ctx, const std::string& key, const Color& fallback) {
    const auto& palette = ctx.palette.get();
    const auto it = palette.find(key);
    if (it == palette.end()) {
        return fallback;
    }
    return it->second;
}

Color ResolveColor(const Context& ctx, const std::string& key) {
    return ResolveColor(ctx, key, Private::ToColor(ImGui::GetStyleColorVec4(ImGuiCol_Text)));
}

}  // namespace Jetstream::Sakura
