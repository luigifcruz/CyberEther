#ifndef JETSTREAM_TESTS_CORE_SAKURA_TYPIST_HH
#define JETSTREAM_TESTS_CORE_SAKURA_TYPIST_HH

#include "harness.hh"
#include "fonts.hh"
#include "component_harness.hh"

#include <functional>
#include <optional>
#include <string>

namespace SakuraTest {

struct Typist {
    SakuraTest::HeadlessUi ui{1.0f, ImVec2(800.0f, 600.0f)};
    MonoFontWindow window;
    Jetstream::Sakura::Context ctx = MakeContext();
    std::function<void()> place;
    Jetstream::Sakura::Component* root = nullptr;

    Typist() {
        ctx.render = &window;
    }

    void frame() {
        ui.frame([&] { place(); });
    }

    void type(const std::string& text) {
        Type(*root, text);
        frame();
    }

    void press(Jetstream::KeyCode key, Jetstream::KeyModifiers modifiers = {}) {
        Tap(*root, key, modifiers);
        frame();
    }

    void blur() {
        ClickFocus(*root, std::nullopt);
        frame();
    }
};

}  // namespace SakuraTest

#endif  // JETSTREAM_TESTS_CORE_SAKURA_TYPIST_HH
