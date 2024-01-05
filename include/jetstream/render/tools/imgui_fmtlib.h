// dear imgui: wrappers for C++ standard library (STL) types (std::string, etc.)
// This is also an example of how you may wrap your own similar types.

#pragma once

#include <fmt/format.h>

namespace ImGui
{
    #define TextFormatted(text, ...) TextUnformatted(fmt::format(text, ##__VA_ARGS__).c_str())
}
