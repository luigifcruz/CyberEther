#pragma once

#include "jetstream/render/tools/imgui.h"

namespace ImGui {

inline ImDrawCallback PremultipliedAlphaCallback = nullptr;

inline void RegisterPremultipliedAlphaCallback(ImDrawCallback bind) {
    PremultipliedAlphaCallback = bind;
}

inline void UnregisterPremultipliedAlphaCallback(ImDrawCallback bind) {
    if (PremultipliedAlphaCallback == bind) {
        PremultipliedAlphaCallback = nullptr;
    }
}

}  // namespace ImGui
