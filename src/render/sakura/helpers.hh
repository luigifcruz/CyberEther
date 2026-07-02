#ifndef JETSTREAM_RENDER_SAKURA_HELPERS_HH
#define JETSTREAM_RENDER_SAKURA_HELPERS_HH

#include "metrics.hh"
#include "palette.hh"

#include "imgui.h"
#include "jetstream/render/tools/imgui_internal.h"
#include "jetstream/render/tools/imgui_stdlib.h"
#include "jetstream/render/tools/imgui_fmtlib.h"
#include "jetstream/render/tools/imgui_icons_ext.hh"
#include "jetstream/render/tools/imgui_notify_ext.h"
#include "jetstream/render/tools/imnodes.h"

#include <jetstream/types.hh>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Jetstream::Sakura::Private {

inline ImVec2 ToImVec2(const Extent2D<F32>& value) {
    return ImVec2(value.x, value.y);
}

inline Extent2D<F32> ToExtent2D(const ImVec2& value) {
    return {value.x, value.y};
}

inline ImVec4 ToImVec4(const ColorRGBA<F32>& value) {
    return ImVec4(value.r, value.g, value.b, value.a);
}

inline ColorRGBA<F32> ToColor(const ImVec4& value) {
    return {value.x, value.y, value.z, value.w};
}

inline ImFont* NativeFont(const FontHandle handle) {
    return static_cast<ImFont*>(handle.native);
}

inline FontHandle ToFontHandle(ImFont* font) {
    return {font};
}

inline ImVec4 ImColor(const Context& ctx, const std::string& key, const ImVec4& fallback) {
    return ToImVec4(Sakura::ResolveColor(ctx, key, ToColor(fallback)));
}

inline ImVec4 ImColor(const Context& ctx, const std::string& key) {
    return ToImVec4(Sakura::ResolveColor(ctx, key));
}

}  // namespace Jetstream::Sakura::Private

#endif  // JETSTREAM_RENDER_SAKURA_HELPERS_HH
