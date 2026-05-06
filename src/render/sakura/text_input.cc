#include <jetstream/render/sakura/text_input.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct TextInput::Impl {
    Config config;
};

TextInput::TextInput() {
    this->impl = std::make_unique<Impl>();
}

TextInput::~TextInput() = default;
TextInput::TextInput(TextInput&&) noexcept = default;
TextInput& TextInput::operator=(TextInput&&) noexcept = default;

bool TextInput::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void TextInput::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    std::string value = config.value;
    ImGui::PushID(config.id.c_str());
    ImGui::SetNextItemWidth(-FLT_MIN);
    if (config.focus && !ImGui::IsAnyItemActive() && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId)) {
        ImGui::SetKeyboardFocusHere();
    }

    bool changed = false;
    if (!config.focusOutline) {
        ImGui::PushStyleColor(ImGuiCol_NavCursor, Private::ImColor(ctx, "transparent"));
    }
    const ImGuiInputTextFlags flags = config.submit == Submit::OnEdit ? ImGuiInputTextFlags_None
                                                                       : ImGuiInputTextFlags_EnterReturnsTrue;
    if (config.hint.empty()) {
        changed = ImGui::InputText("##input", &value, flags);
    } else {
        changed = ImGui::InputTextWithHint("##input", config.hint.c_str(), &value, flags);
    }
    if (!config.focusOutline) {
        ImGui::PopStyleColor();
    }
    if (config.submit == Submit::OnCommit && ImGui::IsItemDeactivatedAfterEdit()) {
        changed = true;
    }

    if (changed && config.onChange) {
        config.onChange(value);
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
