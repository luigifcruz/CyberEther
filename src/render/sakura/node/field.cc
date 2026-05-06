#include <jetstream/render/sakura/node/field.hh>

#include <jetstream/render/sakura/divider.hh>
#include <jetstream/render/sakura/text.hh>
#include <jetstream/render/sakura/tooltip.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeField::Impl {
    Config config;
    Text label;
    Tooltip helpTooltip;
    Text helpText;
    Divider divider;
};

NodeField::NodeField() {
    this->impl = std::make_unique<Impl>();
}

NodeField::~NodeField() = default;
NodeField::NodeField(NodeField&&) noexcept = default;
NodeField& NodeField::operator=(NodeField&&) noexcept = default;

bool NodeField::update(Config config) {
    auto& impl = *this->impl;
    impl.config = std::move(config);
    impl.label.update({
        .id = impl.config.id + "Label",
        .str = impl.config.label,
        .tone = Text::Tone::Secondary,
        .scale = 0.75f,
    });
    impl.helpTooltip.update({
        .id = impl.config.id + "HelpTooltip",
    });
    impl.helpText.update({
        .id = impl.config.id + "HelpText",
        .str = impl.config.help,
        .wrapped = true,
    });
    impl.divider.update({
        .id = impl.config.id + "Divider",
    });
    return true;
}

void NodeField::render(const Context& ctx, Child child) const {
    const auto& impl = *this->impl;
    const auto& config = impl.config;

    if (config.title && !config.label.empty()) {
        if (config.background) {
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() - Scale(ctx, 4.0f));
            const ImVec2 pos = ImGui::GetCursorScreenPos();
            const F32 height = ImGui::GetTextLineHeight() + Scale(ctx, 12.0f);
            const ImVec2 min(pos.x, pos.y - Scale(ctx, 2.0f));
            const ImVec2 max(pos.x + ImGui::GetContentRegionAvail().x, pos.y + height);
            ImGui::GetWindowDrawList()->AddRectFilled(min,
                                                      max,
                                                      ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "card")),
                                                      ImGui::GetStyle().FrameRounding);
        }
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Scale(ctx, 6.0f));
        impl.label.render(ctx);
        if (!config.help.empty()) {
            impl.helpTooltip.render(ctx, [this](const Context& ctx) {
                this->impl->helpText.render(ctx);
            });
        }
        ImGui::SetCursorPosY(ImGui::GetCursorPosY() - Scale(ctx, config.background ? 8.0f : 4.0f));
    }

    if (child) {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, Private::ImColor(ctx, "card"));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, Private::ImColor(ctx, "frame_bg_hovered"));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, Private::ImColor(ctx, "frame_bg_active"));
        child(ctx);
        ImGui::PopStyleColor(3);
    }

    if (config.divider) {
        impl.divider.render(ctx);
    }
}

}  // namespace Jetstream::Sakura
