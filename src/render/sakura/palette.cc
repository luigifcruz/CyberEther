#include "context.hh"

namespace Jetstream::Sakura {

const Palette& EmptyPalette() {
    static const Palette palette;
    return palette;
}

}  // namespace Jetstream::Sakura
