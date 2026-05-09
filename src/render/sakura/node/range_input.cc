#include <jetstream/render/sakura/node/range_input.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeRangeInput::Impl {
    Config config;
};

NodeRangeInput::NodeRangeInput() {
    this->impl = std::make_unique<Impl>();
}

NodeRangeInput::~NodeRangeInput() = default;
NodeRangeInput::NodeRangeInput(NodeRangeInput&&) noexcept = default;
NodeRangeInput& NodeRangeInput::operator=(NodeRangeInput&&) noexcept = default;

bool NodeRangeInput::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void NodeRangeInput::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    const ImU32 bgColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "card"));
    const ImU32 unitColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "text_secondary"));
    const F32 pad = Scale(ctx, 6.0f);
    const F32 rounding = ImGui::GetStyle().FrameRounding;
    const F32 availWidth = ImGui::GetContentRegionAvail().x;
    const F32 frameHeight = ImGui::GetTextLineHeight() + Scale(ctx, 6.0f);

    F32 value = config.value;
    ImGui::PushID(config.id.c_str());
    ImGui::InvisibleButton("##range", ImVec2(availWidth, frameHeight));

    const ImVec2 rectMin = ImGui::GetItemRectMin();
    const ImVec2 rectMax = ImGui::GetItemRectMax();
    ImGui::GetWindowDrawList()->AddRectFilled(rectMin, rectMax, bgColor, rounding);

    const F32 range = config.max - config.min;
    F32 fraction = (range != 0.0f) ? (value - config.min) / range : 0.0f;
    fraction = std::clamp(fraction, 0.0f, 1.0f);
    bool changed = false;

    if (ImGui::IsItemActive()) {
        const F32 mouseX = ImGui::GetIO().MousePos.x;
        const F32 newFraction = std::clamp((mouseX - rectMin.x) / (rectMax.x - rectMin.x), 0.0f, 1.0f);
        const F32 newValue = config.min + newFraction * range;
        if (newValue != value) {
            value = config.integer ? std::round(newValue) : newValue;
            fraction = newFraction;
            changed = true;
        }
    }

    const F32 sliderWidth = (rectMax.x - rectMin.x) * fraction;
    const ImU32 sliderColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "frame_bg_hovered"));
    const F32 knobWidth = ImMax(Scale(ctx, 8.0f), rounding * 2.0f);
    const F32 knobX = std::clamp(rectMin.x + sliderWidth - knobWidth * 0.5f, rectMin.x, rectMax.x - knobWidth);
    ImGui::GetWindowDrawList()->AddRectFilled(rectMin, ImVec2(knobX + knobWidth, rectMax.y), sliderColor, rounding);

    const ImU32 knobColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "frame_bg_active"));
    ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(knobX, rectMin.y),
                                              ImVec2(knobX + knobWidth, rectMax.y),
                                              knobColor,
                                              rounding);

    const std::string valueText = config.integer ? jst::fmt::format("{}", static_cast<U64>(value))
                                                 : jst::fmt::format("{:.0f}", value);
    const ImVec2 textPos(rectMin.x + pad, rectMin.y + (frameHeight - ImGui::GetFontSize()) * 0.5f);
    ImGui::GetWindowDrawList()->AddText(textPos, ImGui::GetColorU32(ImGuiCol_Text), valueText.c_str());
    if (!config.unit.empty()) {
        const ImVec2 unitTextSize = ImGui::CalcTextSize(config.unit.c_str());
        const ImVec2 unitTextPos(rectMax.x - unitTextSize.x - Scale(ctx, 3.0f),
                                 rectMin.y + (frameHeight - unitTextSize.y) * 0.5f);
        ImGui::GetWindowDrawList()->AddText(unitTextPos, unitColor, config.unit.c_str());
    }

    if (changed && config.onChange) {
        config.onChange(value);
    }
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
