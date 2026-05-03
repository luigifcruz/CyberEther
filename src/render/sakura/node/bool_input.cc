#include <jetstream/render/sakura/node/bool_input.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeBoolInput::Impl {
    Config config;
};

NodeBoolInput::NodeBoolInput() {
    this->impl = std::make_unique<Impl>();
}

NodeBoolInput::~NodeBoolInput() = default;
NodeBoolInput::NodeBoolInput(NodeBoolInput&&) noexcept = default;
NodeBoolInput& NodeBoolInput::operator=(NodeBoolInput&&) noexcept = default;

bool NodeBoolInput::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void NodeBoolInput::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    const std::string displayText = config.value ? "Enabled" : "Disabled";
    const char* icon = config.value ? ICON_FA_TOGGLE_ON : ICON_FA_TOGGLE_OFF;
    const F32 height = ImGui::GetTextLineHeight() + Scale(ctx, 6.0f);
    const F32 pad = Scale(ctx, 6.0f);
    const F32 iconFontSize = ImGui::GetFontSize() * 0.8f;
    const F32 iconWidth = ImGui::GetFont()->CalcTextSizeA(iconFontSize, FLT_MAX, 0.0f, icon).x;

    ImGui::PushID(config.id.c_str());
    const bool changed = ImGui::InvisibleButton("##bool", ImVec2(ImGui::GetContentRegionAvail().x, height));
    if (changed && config.onChange) {
        config.onChange(!config.value);
    }

    const ImVec2 rectMin = ImGui::GetItemRectMin();
    const ImVec2 rectMax = ImGui::GetItemRectMax();
    const ImVec2 textSize = ImGui::CalcTextSize(displayText.c_str());
    const ImVec2 textPos(rectMin.x + pad, rectMin.y + (height - textSize.y) * 0.5f);
    const F32 iconX = rectMax.x - pad - iconWidth;
    const F32 iconY = rectMin.y + (height - iconFontSize) * 0.5f;

    if (ImGui::IsItemHovered()) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }
    ImGui::GetWindowDrawList()->AddRectFilled(rectMin,
                                              rectMax,
                                              ImGui::GetColorU32(ImGuiCol_FrameBg),
                                              ImGui::GetStyle().FrameRounding);
    ImGui::GetWindowDrawList()->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), displayText.c_str());
    const ImU32 iconColor = config.value ? ImGui::GetColorU32(ImVec4(0.4f, 0.8f, 0.4f, 1.0f))
                                         : ImGui::GetColorU32(ImGuiCol_TextDisabled);
    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(), iconFontSize, ImVec2(iconX, iconY), iconColor, icon);
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
