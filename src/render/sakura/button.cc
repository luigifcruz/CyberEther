#include <jetstream/render/sakura/button.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct Button::Impl {
    Config config;
};

Button::Button() {
    this->impl = std::make_unique<Impl>();
}

Button::~Button() = default;
Button::Button(Button&&) noexcept = default;
Button& Button::operator=(Button&&) noexcept = default;

bool Button::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void Button::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    ImGui::PushID(config.id.c_str());

    I32 styleColorCount = 0;
    const bool customColors = !config.colorKey.empty() ||
                              !config.hoveredColorKey.empty() ||
                              !config.activeColorKey.empty();
    if (config.variant == Variant::Action || customColors) {
        ImGui::PushStyleColor(ImGuiCol_Button, Private::ImColor(ctx, config.colorKey.empty() ? "action_btn" : config.colorKey));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, Private::ImColor(ctx, config.hoveredColorKey.empty() ? "action_btn_hovered" : config.hoveredColorKey));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, Private::ImColor(ctx, config.activeColorKey.empty() ? "action_btn_active" : config.activeColorKey));
        styleColorCount += 3;
    } else if (config.variant == Variant::Destructive) {
        const ImVec4 error = Private::ImColor(ctx, "error_red", ImVec4(1.00f, 0.00f, 0.00f, 1.00f));
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(error.x * 0.60f, error.y * 0.60f, error.z * 0.60f, error.w));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(error.x * 0.70f, error.y * 0.70f, error.z * 0.70f, error.w));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(error.x * 0.50f, error.y * 0.50f, error.z * 0.50f, error.w));
        styleColorCount += 3;
    }

    if (!config.textColorKey.empty()) {
        ImGui::PushStyleColor(ImGuiCol_Text, Private::ImColor(ctx, config.textColorKey));
        styleColorCount += 1;
    }

    if (config.disabled) {
        ImGui::BeginDisabled();
    }

    if (ImGui::Button(config.str.c_str(), Private::ToImVec2(Scale(ctx, config.size)))) {
        if (config.onClick) {
            config.onClick();
        }
    }

    if (config.disabled) {
        ImGui::EndDisabled();
    }

    if (styleColorCount > 0) {
        ImGui::PopStyleColor(styleColorCount);
    }

    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
