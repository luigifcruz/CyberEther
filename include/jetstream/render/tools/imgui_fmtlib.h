// dear imgui: wrappers for C++ standard library (STL) types (std::string, etc.)
// This is also an example of how you may wrap your own similar types.

#pragma once

#include "jetstream/logger.hh"

namespace ImGui
{
    #define TextFormatted(text, ...) TextUnformatted(jst::fmt::format(text, ##__VA_ARGS__).c_str())
}
