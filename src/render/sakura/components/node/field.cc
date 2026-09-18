#include <jetstream/render/sakura/components/node/field.hh>

#include <jetstream/render/sakura/components/divider.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/tooltip.hh>

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
    impl->config = std::move(config);
    impl->label.update({
        .id = impl->config.id + "Label",
        .str = impl->config.label,
        .tone = Text::Tone::Secondary,
        .scale = 0.75f,
    });
    impl->helpTooltip.update({
        .id = impl->config.id + "HelpTooltip",
    });
    impl->helpText.update({
        .id = impl->config.id + "HelpText",
        .str = impl->config.help,
        .wrapped = true,
    });
    impl->divider.update({
        .id = impl->config.id + "Divider",
    });
    return true;
}

void NodeField::render(const Context& ctx, Child child) const {
    const auto& config = impl->config;

    if (config.title && !config.label.empty()) {
        if (config.background) {
            const ImVec2 pos = ImGui::GetCursorScreenPos();
            const F32 height = ImGui::GetTextLineHeight() + Scale(ctx, 14.0f);
            const ImVec2 max(pos.x + ImGui::GetContentRegionAvail().x, pos.y + height);
            ImGui::GetWindowDrawList()->AddRectFilled(pos,
                                                      max,
                                                      ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, "card")),
                                                      ImGui::GetStyle().FrameRounding);
            ImGui::SetCursorPosY(ImGui::GetCursorPosY() + Scale(ctx, 2.0f));
        }
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + Scale(ctx, 6.0f));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,
                            ImVec2(ImGui::GetStyle().ItemSpacing.x,
                                   Scale(ctx, config.background ? 0.0f : 4.0f)));
        impl->label.render(ctx);
        ImGui::PopStyleVar();
        if (!config.help.empty()) {
            impl->helpTooltip.render(ctx, [this](const Context& ctx) {
                this->impl->helpText.render(ctx);
            });
        }
    }

    if (child) {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, Private::ImColor(ctx, "card"));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, Private::ImColor(ctx, "frame_bg_hovered"));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, Private::ImColor(ctx, "frame_bg_active"));
        child(ctx);
        ImGui::PopStyleColor(3);
    }

    if (config.divider) {
        impl->divider.render(ctx);
    }
}

}  // namespace Jetstream::Sakura
