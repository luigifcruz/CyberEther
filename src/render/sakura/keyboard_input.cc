#include <jetstream/render/sakura/keyboard_input.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct KeyboardInput::Impl {
    Config config;

    bool modifierPressed(Modifier modifier) const {
        switch (modifier) {
            case Modifier::None:
                return true;
            case Modifier::CommandOrControl: {
                const ImGuiIO& io = ImGui::GetIO();
                return (io.KeyMods & ImGuiMod_Super) != 0 || (io.KeyMods & ImGuiMod_Ctrl) != 0;
            }
        }
        return false;
    }

    bool pressed(Key key) const {
        switch (key) {
            case Key::Down: return ImGui::IsKeyPressed(ImGuiKey_DownArrow, config.repeat);
            case Key::Up: return ImGui::IsKeyPressed(ImGuiKey_UpArrow, config.repeat);
            case Key::Submit:
                return ImGui::IsKeyPressed(ImGuiKey_Enter, config.repeat) ||
                       ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, config.repeat);
            case Key::N: return ImGui::IsKeyPressed(ImGuiKey_N, config.repeat);
            case Key::O: return ImGui::IsKeyPressed(ImGuiKey_O, config.repeat);
            case Key::S: return ImGui::IsKeyPressed(ImGuiKey_S, config.repeat);
            case Key::W: return ImGui::IsKeyPressed(ImGuiKey_W, config.repeat);
            case Key::I: return ImGui::IsKeyPressed(ImGuiKey_I, config.repeat);
            case Key::Comma: return ImGui::IsKeyPressed(ImGuiKey_Comma, config.repeat);
        }
        return false;
    }
};

KeyboardInput::KeyboardInput() {
    this->impl = std::make_unique<Impl>();
}

KeyboardInput::~KeyboardInput() = default;
KeyboardInput::KeyboardInput(KeyboardInput&&) noexcept = default;
KeyboardInput& KeyboardInput::operator=(KeyboardInput&&) noexcept = default;

bool KeyboardInput::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void KeyboardInput::render(const Context& ctx) const {
    (void)ctx;
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());
    for (const auto& binding : config.bindings) {
        if (this->impl->modifierPressed(binding.modifier) && this->impl->pressed(binding.key) && binding.onPressed) {
            binding.onPressed();
        }
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
