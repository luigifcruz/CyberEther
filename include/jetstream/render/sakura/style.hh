#ifndef JETSTREAM_RENDER_SAKURA_STYLE_HH
#define JETSTREAM_RENDER_SAKURA_STYLE_HH

#include <jetstream/types.hh>

namespace Jetstream::Sakura {

struct Rounding {
    static constexpr F32 Control = 8.0f;
    static constexpr F32 Card = 12.0f;
    static constexpr F32 Popup = 18.0f;
};

struct Spacing {
    static constexpr F32 XSmall = 4.0f;
    static constexpr F32 Small = 8.0f;
    static constexpr F32 Medium = 12.0f;
    static constexpr F32 Large = 16.0f;
    static constexpr F32 XLarge = 24.0f;
    static constexpr F32 XXLarge = 32.0f;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_STYLE_HH
