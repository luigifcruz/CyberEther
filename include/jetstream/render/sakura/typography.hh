#ifndef JETSTREAM_RENDER_SAKURA_TYPOGRAPHY_HH
#define JETSTREAM_RENDER_SAKURA_TYPOGRAPHY_HH

#include <jetstream/types.hh>

namespace Jetstream::Sakura {

struct Typography {
    static constexpr F32 FontSize = 15.0f;
    static constexpr F32 ChromeScale = 0.90f;
    static constexpr const char* MonoFont = "default_mono";
    static constexpr const char* BodyFont = "default_body";
    static constexpr F32 CodeLineHeight = 1.15f;
    static constexpr F32 BodyLineHeight = 1.25f;
    static constexpr F32 HeadingScale[3] = {1.5f, 1.3f, 1.15f};
    static constexpr F32 ChromeH1Scale = 1.15f;
    static constexpr F32 ChromeH2Scale = 1.10f;
    static constexpr F32 ChromeBoldScale = 1.04f;
    static constexpr F32 DisplayScale = 1.15f;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_TYPOGRAPHY_HH
