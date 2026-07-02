#include <jetstream/render/sakura/components/node/title.hh>

#include <jetstream/render/sakura/components/divider.hh>
#include <jetstream/render/sakura/components/hstack.hh>
#include <jetstream/render/sakura/components/text.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeTitle::Impl {
    Config config;
    Text title;
    HStack diagnosticHeader;
    Text diagnosticHeaderIcon;
    Text diagnosticHeaderLabel;
    Divider diagnosticDivider;
    Text diagnosticMessage;

    std::string diagnosticColorKey() const {
        return config.diagnostic.state == Node::State::Error ? "node_outline_error"
                                                             : "node_outline_pending";
    }
};

NodeTitle::NodeTitle() {
    this->impl = std::make_unique<Impl>();
}

NodeTitle::~NodeTitle() = default;
NodeTitle::NodeTitle(NodeTitle&&) noexcept = default;
NodeTitle& NodeTitle::operator=(NodeTitle&&) noexcept = default;

bool NodeTitle::update(Config config) {
    impl->config = std::move(config);
    const std::string id = "NodeTitle" + impl->config.title;
    impl->title.update({
        .id = id + "Text",
        .str = impl->config.title,
        .align = Text::Align::Center,
        .scale = impl->config.titleScale,
    });
    impl->diagnosticHeader.update({
        .id = id + "DiagnosticHeader",
        .spacing = 4.0f,
    });
    impl->diagnosticHeaderIcon.update({
        .id = id + "DiagnosticHeaderIcon",
        .str = ICON_FA_TRIANGLE_EXCLAMATION,
        .colorKey = impl->diagnosticColorKey(),
    });
    impl->diagnosticHeaderLabel.update({
        .id = id + "DiagnosticHeaderLabel",
        .str = "Diagnostic",
    });
    impl->diagnosticDivider.update({
        .id = id + "DiagnosticDivider",
        .spacing = 0.0f,
    });
    impl->diagnosticMessage.update({
        .id = id + "DiagnosticMessage",
        .str = impl->config.diagnostic.message,
    });
    return true;
}

void NodeTitle::render(const Context& ctx) const {
    const auto& config = impl->config;

    ImNodes::BeginNodeTitleBar();

    const bool hasDiagnostic = config.diagnostic.state != Node::State::Normal && !config.diagnostic.message.empty();
    const ImVec2 titleStartScreen = ImGui::GetCursorScreenPos();
    const F32 availWidth = ImGui::GetContentRegionAvail().x;
    ImVec2 diagnosticIconMin(0.0f, 0.0f);
    ImVec2 diagnosticIconMax(0.0f, 0.0f);

    impl->title.render(ctx);

    if (hasDiagnostic) {
        const ImVec2 titleMin = ImGui::GetItemRectMin();
        const ImVec2 titleMax = ImGui::GetItemRectMax();
        const ImVec2 iconSize = ImGui::CalcTextSize(ICON_FA_SKULL);
        diagnosticIconMin.x = titleStartScreen.x + ImMax(0.0f, availWidth - iconSize.x);
        diagnosticIconMin.y = titleMin.y + ImMax(0.0f, (titleMax.y - titleMin.y - iconSize.y) * 0.5f);
        diagnosticIconMax = ImVec2(diagnosticIconMin.x + iconSize.x,
                                   diagnosticIconMin.y + iconSize.y);
    }

    ImNodes::EndNodeTitleBar();

    if (hasDiagnostic) {
        const ImU32 iconColor = ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, impl->diagnosticColorKey()));
        ImGui::GetWindowDrawList()->AddText(diagnosticIconMin, iconColor, ICON_FA_SKULL);

        if (ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
            ImGui::IsMouseHoveringRect(diagnosticIconMin, diagnosticIconMax)) {
            ImGui::BeginTooltip();
            ImGui::PushTextWrapPos(Scale(ctx, 420.0f));
            impl->diagnosticHeader.render(ctx, {
                [&](const Context& ctx) {
                    impl->diagnosticHeaderIcon.render(ctx);
                },
                [&](const Context& ctx) {
                    impl->diagnosticHeaderLabel.render(ctx);
                },
            });
            impl->diagnosticDivider.render(ctx);
            impl->diagnosticMessage.render(ctx);
            ImGui::PopTextWrapPos();
            ImGui::EndTooltip();
        }
    }
}

}  // namespace Jetstream::Sakura
