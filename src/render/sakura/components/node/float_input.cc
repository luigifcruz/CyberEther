#include <jetstream/render/sakura/components/node/float_input.hh>

#include "base.hh"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <string>
#include <utility>

namespace Jetstream::Sakura {

namespace {

F64 Unscale(const F64 shown, const F64 scale) {
    const F64 decimals = std::round(std::log10(scale));
    if (!std::isfinite(shown) || !std::isfinite(scale) || scale <= 0.0 || std::pow(10.0, decimals) != scale) {
        return shown * scale;
    }

    std::string number = jst::fmt::format("{}", shown);
    long exponent = 0;
    if (const auto mark = number.find_first_of("eE"); mark != std::string::npos) {
        errno = 0;
        exponent = std::strtol(number.c_str() + mark + 1, nullptr, 10);
        if (errno == ERANGE || !std::in_range<I32>(exponent)) {
            return shown * scale;
        }
        number.resize(mark);
    }
    exponent += static_cast<long>(decimals);

    errno = 0;
    const F64 value = std::strtod((number + "e" + std::to_string(exponent)).c_str(), nullptr);
    if (errno == ERANGE || !std::isfinite(value)) {
        return shown * scale;
    }
    return value;
}

}  // namespace

struct NodeFloatInput::Impl {
    Config config;
};

NodeFloatInput::NodeFloatInput() {
    this->impl = std::make_unique<Impl>();
}

NodeFloatInput::~NodeFloatInput() = default;
NodeFloatInput::NodeFloatInput(NodeFloatInput&&) noexcept = default;
NodeFloatInput& NodeFloatInput::operator=(NodeFloatInput&&) noexcept = default;

bool NodeFloatInput::update(Config config) {
    this->impl->config = std::move(config);
    return true;
}

void NodeFloatInput::render(const Context& ctx) const {
    const auto& config = this->impl->config;

    F64 shown = config.value / config.scale;
    F64 shownStep = config.step.value_or(0.0) / config.scale;
    const auto commit = [&]() {
        const F64 next = Unscale(shown, config.scale);
        if (next != config.value && config.onChange) {
            config.onChange(next);
        }
    };

    char fmt[16];
    snprintf(fmt, sizeof(fmt), "%%.%df", config.precision);

    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(Scale(ctx, 6.0f), Scale(ctx, 3.0f)));
    if (!config.step.has_value()) {
        ImGui::SetNextItemWidth(-FLT_MIN);
        if (ImGui::InputDouble("##float", &shown, 0.0, 0.0, fmt, ImGuiInputTextFlags_EnterReturnsTrue)) {
            commit();
        }
        if (!config.unit.empty()) {
            const ImVec2 inputPos = ImGui::GetItemRectMin();
            const ImVec2 inputSize = ImGui::GetItemRectSize();
            const ImVec2 textSize = ImGui::CalcTextSize(config.unit.c_str());
            const ImVec2 textPos(inputPos.x + inputSize.x - textSize.x - Scale(ctx, 3.0f),
                                 inputPos.y + (inputSize.y - textSize.y) * 0.5f);
            ImGui::GetWindowDrawList()->AddText(textPos,
                                                ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "text_secondary")),
                                                config.unit.c_str());
        }
        ImGui::PopStyleVar();
        ImGui::PopID();
        return;
    }

    const F32 availWidth = ImGui::GetContentRegionAvail().x;
    const F32 btnHeight = ImGui::GetTextLineHeight() + Scale(ctx, 4.0f);
    const ImU32 bgColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "card"));
    const ImU32 btnColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "text_secondary"));
    const ImU32 btnActiveColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "text_primary"));
    const F32 rounding = ImGui::GetStyle().FrameRounding;
    const F32 btnWidth = availWidth * 0.2f;
    const F32 stepInputWidth = availWidth - btnWidth * 2.0f;

    const auto drawCenteredLabel = [&](const char* label,
                                       const ImVec2& min,
                                       const ImVec2& max,
                                       const bool hovered) {
        const ImVec2 textSize = ImGui::CalcTextSize(label);
        ImGui::GetWindowDrawList()->AddText(ImVec2((min.x + max.x) * 0.5f - textSize.x * 0.5f,
                                                   (min.y + max.y) * 0.5f - textSize.y * 0.5f),
                                            hovered ? btnActiveColor : btnColor,
                                            label);
    };
    const auto renderStepButton = [&](const char* id, const char* label, const F64 delta) {
        ImGui::InvisibleButton(id, ImVec2(btnWidth, btnHeight));
        const bool hovered = ImGui::IsItemHovered();
        if (hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        }
        if (ImGui::IsItemDeactivated() && config.onChange) {
            config.onChange(config.value + delta);
        }
        drawCenteredLabel(label, ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), hovered);
    };

    const ImVec2 cellPos = ImGui::GetCursorScreenPos();
    const F32 inputHeight = ImGui::GetFrameHeight();
    const F32 totalHeight = inputHeight + btnHeight;
    ImGui::GetWindowDrawList()->AddRectFilled(cellPos,
                                              ImVec2(cellPos.x + availWidth, cellPos.y + totalHeight),
                                              bgColor,
                                              rounding);

    ImGui::SetNextItemWidth(availWidth);
    if (ImGui::InputDouble("##float", &shown, 0.0, 0.0, fmt, ImGuiInputTextFlags_EnterReturnsTrue)) {
        commit();
    }
    if (!config.unit.empty()) {
        const ImVec2 inputPos = ImGui::GetItemRectMin();
        const ImVec2 inputSize = ImGui::GetItemRectSize();
        const ImVec2 textSize = ImGui::CalcTextSize(config.unit.c_str());
        const ImVec2 textPos(inputPos.x + inputSize.x - textSize.x - Scale(ctx, 3.0f),
                             inputPos.y + (inputSize.y - textSize.y) * 0.5f);
        ImGui::GetWindowDrawList()->AddText(textPos,
                                            ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "text_secondary")),
                                            config.unit.c_str());
    }

    const ImVec2 inputMax = ImGui::GetItemRectMax();
    const ImVec2 btnRowPos(cellPos.x, inputMax.y);
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));
    ImGui::SetCursorScreenPos(btnRowPos);
    renderStepButton("-##float_step_minus", "-", -config.step.value_or(0.0));

    ImGui::SameLine(0.0f, 0.0f);
    const ImGuiID stepEditId = ImGui::GetID("##step_editing");
    const ImGuiID stepFocusId = ImGui::GetID("##step_focus");
    ImGuiStorage* storage = ImGui::GetStateStorage();
    if (!storage->GetBool(stepEditId, false)) {
        ImGui::InvisibleButton("##step_toggle", ImVec2(stepInputWidth, btnHeight));
        const bool hovered = ImGui::IsItemHovered();
        if (hovered) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        }
        if (ImGui::IsItemDeactivated()) {
            storage->SetBool(stepEditId, true);
            storage->SetBool(stepFocusId, true);
        }
        drawCenteredLabel("Step Size", ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), hovered);
    } else {
        ImGui::SetNextItemWidth(stepInputWidth);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding,
                            ImVec2(ImGui::GetStyle().FramePadding.x,
                                   (btnHeight - ImGui::GetTextLineHeight()) * 0.5f));
        ImGui::PushStyleColor(ImGuiCol_Text, Private::ImColor(ctx, "text_secondary"));
        if (storage->GetBool(stepFocusId, false)) {
            ImGui::SetKeyboardFocusHere();
            storage->SetBool(stepFocusId, false);
        }
        if (ImGui::InputDouble("##float_step", &shownStep, 0.0, 0.0, fmt, ImGuiInputTextFlags_EnterReturnsTrue)) {
            if (config.onStepChange) {
                config.onStepChange(Unscale(shownStep, config.scale));
            }
        }
        if (ImGui::IsItemDeactivated()) {
            storage->SetBool(stepEditId, false);
            storage->SetBool(stepFocusId, false);
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleVar();
    }

    ImGui::SameLine(0.0f, 0.0f);
    renderStepButton("+##float_step_plus", "+", config.step.value_or(0.0));
    ImGui::PopStyleVar();

    ImGui::SetCursorScreenPos(ImVec2(cellPos.x, cellPos.y + totalHeight));
    ImGui::Dummy(ImVec2(0.0f, 0.0f));
    ImGui::PopStyleVar();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
