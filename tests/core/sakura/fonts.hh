#ifndef JETSTREAM_TESTS_CORE_SAKURA_FONTS_HH
#define JETSTREAM_TESTS_CORE_SAKURA_FONTS_HH

#include "../../../resources/fonts/compressed_jbmm.hh"
#include "harness.hh"

namespace SakuraTest {

struct MonoFontWindow : SakuraTest::FontWindow {
    MonoFontWindow() {
        load("default_mono", jbmm_compressed_data);
    }
};

}  // namespace SakuraTest

#endif  // JETSTREAM_TESTS_CORE_SAKURA_FONTS_HH
