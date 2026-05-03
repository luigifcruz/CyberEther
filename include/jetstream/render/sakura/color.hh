#pragma once

#include <jetstream/types.hh>

#include <string>
#include <unordered_map>

namespace Jetstream::Sakura {

struct Context;

struct Color {
    F32 x = 0.0f;
    F32 y = 0.0f;
    F32 z = 0.0f;
    F32 w = 1.0f;
};

using Palette = std::unordered_map<std::string, Color>;

const Palette& EmptyPalette();
Color ResolveColor(const Context& ctx, const std::string& key, const Color& fallback);
Color ResolveColor(const Context& ctx, const std::string& key);

}  // namespace Jetstream::Sakura
