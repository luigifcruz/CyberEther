#include "palette.hh"

#include "helpers.hh"

namespace Jetstream::Sakura {

const Palette& EmptyPalette() {
    static const Palette palette;
    return palette;
}

ColorRGBA<F32> ResolveColor(const Context& ctx, const std::string& key, const ColorRGBA<F32>& fallback) {
    const auto& palette = ctx.palette.get();
    const auto it = palette.find(key);
    if (it == palette.end()) {
        return fallback;
    }
    return it->second;
}

ColorRGBA<F32> ResolveColor(const Context& ctx, const std::string& key) {
    return ResolveColor(ctx, key, Private::ToColor(ImGui::GetStyleColorVec4(ImGuiCol_Text)));
}

}  // namespace Jetstream::Sakura
