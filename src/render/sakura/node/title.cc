#include <jetstream/render/sakura/node/title.hh>

#include <jetstream/render/sakura/divider.hh>
#include <jetstream/render/sakura/hstack.hh>
#include <jetstream/render/sakura/text.hh>
#include <jetstream/render/sakura/tooltip.hh>

#include "base.hh"

namespace Jetstream::Sakura {

struct NodeTitle::Impl {
    Config config;
    Text title;
    Text diagnosticIcon;
    HStack diagnosticHeader;
    Text diagnosticHeaderIcon;
    Text diagnosticHeaderLabel;
    Divider diagnosticDivider;
    Tooltip diagnosticTooltip;
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
    auto& impl = *this->impl;
    impl.config = std::move(config);
    const std::string id = "NodeTitle" + impl.config.title;
    impl.title.update({
        .id = id + "Text",
        .str = impl.config.title,
        .align = Text::Align::Center,
        .scale = impl.config.titleScale,
    });
    impl.diagnosticIcon.update({
        .id = id + "DiagnosticIcon",
        .str = ICON_FA_SKULL,
        .colorKey = impl.diagnosticColorKey(),
    });
    impl.diagnosticHeader.update({
        .id = id + "DiagnosticHeader",
        .spacing = 4.0f,
    });
    impl.diagnosticHeaderIcon.update({
        .id = id + "DiagnosticHeaderIcon",
        .str = ICON_FA_TRIANGLE_EXCLAMATION,
        .colorKey = impl.diagnosticColorKey(),
    });
    impl.diagnosticHeaderLabel.update({
        .id = id + "DiagnosticHeaderLabel",
        .str = "Diagnostic",
    });
    impl.diagnosticDivider.update({
        .id = id + "DiagnosticDivider",
        .spacing = 0.0f,
    });
    impl.diagnosticTooltip.update({
        .id = id + "DiagnosticTooltip",
    });
    impl.diagnosticMessage.update({
        .id = id + "DiagnosticMessage",
        .str = impl.config.diagnostic.message,
    });
    return true;
}

void NodeTitle::render(const Context& ctx) const {
    const auto& impl = *this->impl;
    const auto& config = impl.config;

    ImNodes::BeginNodeTitleBar();

    const bool hasDiagnostic = config.diagnostic.state != Node::State::Normal && !config.diagnostic.message.empty();
    const F32 titleStartX = ImGui::GetCursorPosX();
    const F32 availWidth = ImGui::GetContentRegionAvail().x;

    impl.title.render(ctx);

    if (hasDiagnostic) {
        ImGui::SameLine();
        ImGui::SetCursorPosX(titleStartX + availWidth - ImGui::CalcTextSize(ICON_FA_SKULL).x);
        impl.diagnosticIcon.render(ctx);
        impl.diagnosticTooltip.render(ctx, [this](const Context& ctx) {
            const auto& impl = *this->impl;
            impl.diagnosticHeader.render(ctx, {
                [this](const Context& ctx) { this->impl->diagnosticHeaderIcon.render(ctx); },
                [this](const Context& ctx) { this->impl->diagnosticHeaderLabel.render(ctx); },
            });
            impl.diagnosticDivider.render(ctx);
            impl.diagnosticMessage.render(ctx);
        });
    }

    ImNodes::EndNodeTitleBar();
}

}  // namespace Jetstream::Sakura
