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
    MonoFontWindow window;
    Jetstream::Sakura::Context ctx = MakeContext();
    std::function<void()> place;
    Jetstream::Sakura::Component* root = nullptr;

    Typist() {
        ctx.render = &window;
    }

    void frame() {
        window.frame([&] { place(); });
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
