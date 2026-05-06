#include <jetstream/render/sakura/node/path_input.hh>

#include <jetstream/render/sakura/text.hh>
#include <jetstream/render/sakura/tooltip.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodePathInput::Impl {
    Config config;
    Tooltip pathTooltip;
    Text pathTooltipText;
};

NodePathInput::NodePathInput() {
    this->impl = std::make_unique<Impl>();
}

NodePathInput::~NodePathInput() = default;
NodePathInput::NodePathInput(NodePathInput&&) noexcept = default;
NodePathInput& NodePathInput::operator=(NodePathInput&&) noexcept = default;

bool NodePathInput::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.pathTooltip.update({
        .id = impl.config.id + "Tooltip",
    });
    impl.pathTooltipText.update({
        .id = impl.config.id + "TooltipText",
        .str = impl.config.value,
        .wrapped = true,
    });
    return true;
}

void NodePathInput::render(const Context& ctx) const {
    const auto& impl = *this->impl;
    const auto& config = impl.config;

    const ImU32 bgColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "card"));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(Scale(ctx, 6.0f), Scale(ctx, 3.0f)));
    const F32 buttonWidth = ImGui::GetFrameHeight();
    const F32 frameHeight = ImGui::GetFrameHeight();
    const F32 connectorWidth = ImGui::GetStyle().ItemSpacing.x;
    const F32 inputWidth = std::max(0.0f, ImGui::GetContentRegionAvail().x - buttonWidth - connectorWidth);
    std::string value = config.value;

    ImGui::PushID(config.id.c_str());
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::ColorConvertU32ToFloat4(bgColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::ColorConvertU32ToFloat4(bgColor));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::ColorConvertU32ToFloat4(bgColor));

    const ImVec2 fieldRectMin = ImGui::GetCursorScreenPos();
    ImGui::GetWindowDrawList()->AddRectFilled(fieldRectMin,
                                              ImVec2(fieldRectMin.x + inputWidth + connectorWidth + buttonWidth,
                                                     fieldRectMin.y + frameHeight),
                                              bgColor,
                                              ImGui::GetStyle().FrameRounding);

    ImGui::SetNextItemWidth(inputWidth);
    if (ImGui::InputTextWithHint("##path", "Select file...", &value, ImGuiInputTextFlags_EnterReturnsTrue)) {
        if (config.onChange) {
            config.onChange(value);
        }
    }
    if (!config.value.empty()) {
        impl.pathTooltip.render(ctx, [this](const Context& ctx) {
            this->impl->pathTooltipText.render(ctx);
        });
    }

    ImGui::SameLine(0.0f, 0.0f);
    ImGui::Dummy(ImVec2(connectorWidth, frameHeight));
    ImGui::SameLine(0.0f, 0.0f);

    const F32 iconFontSize = ImGui::GetFontSize() * 0.7f;
    const ImVec2 iconSize = ImGui::GetFont()->CalcTextSizeA(iconFontSize, FLT_MAX, 0.0f, ICON_FA_FOLDER_OPEN);
    if (ImGui::Button("##browse", ImVec2(buttonWidth, frameHeight)) && config.onBrowse) {
        config.onBrowse();
    }
    const ImVec2 buttonRectMin = ImGui::GetItemRectMin();
    const ImVec2 buttonRectMax = ImGui::GetItemRectMax();
    const F32 iconX = buttonRectMin.x + ((buttonRectMax.x - buttonRectMin.x) - iconSize.x) * 0.5f;
    const F32 iconY = buttonRectMin.y + (frameHeight - iconFontSize) * 0.5f;
    ImGui::GetWindowDrawList()->AddText(ImGui::GetFont(),
                                        iconFontSize,
                                        ImVec2(iconX, iconY),
                                        ImGui::GetColorU32(ImGuiCol_Text),
                                        ICON_FA_FOLDER_OPEN);
    if (!config.value.empty()) {
        impl.pathTooltip.render(ctx, [this](const Context& ctx) {
            this->impl->pathTooltipText.render(ctx);
        });
    }

    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar();
    ImGui::PopID();
}

}  // namespace Jetstream::Sakura
