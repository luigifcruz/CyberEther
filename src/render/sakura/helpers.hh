#ifndef JETSTREAM_RENDER_SAKURA_HELPERS_HH
#define JETSTREAM_RENDER_SAKURA_HELPERS_HH

#include "context.hh"
#include "metrics.hh"

#include "imgui.h"
#include "jetstream/render/tools/imgui_internal.h"
#include "jetstream/render/tools/imgui_icons_ext.hh"

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

}  // namespace Jetstream::Sakura::Private

#endif  // JETSTREAM_RENDER_SAKURA_HELPERS_HH
