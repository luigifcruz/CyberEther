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
    mutable bool chevronArmed = false;

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
    const bool hasChevron = config.configHasFields && config.onToggleConfigCollapse;
    const ImVec2 titleStartScreen = ImGui::GetCursorScreenPos();
    const F32 availWidth = ImGui::GetContentRegionAvail().x;
    const F32 gap = Scale(ctx, 8.0f);
    const ImVec2 chevronSize = hasChevron
        ? ImGui::CalcTextSize(ICON_FA_CHEVRON_DOWN)
        : ImVec2(0.0f, 0.0f);
    const ImVec2 skullSize = hasDiagnostic
        ? ImGui::CalcTextSize(ICON_FA_SKULL)
        : ImVec2(0.0f, 0.0f);

    F32 reservedRight = 0.0f;
    if (hasChevron) {
        reservedRight += chevronSize.x;
    }
    if (hasChevron && hasDiagnostic) {
        reservedRight += gap;
    }
    if (hasDiagnostic) {
        reservedRight += skullSize.x;
    }
    if (reservedRight > 0.0f) {
        reservedRight += gap;
    }

    const F32 titleAvailWidth = ImMax(0.0f, availWidth - reservedRight);
    const F32 fullCenterSafeWidth = ImMax(0.0f, availWidth - 2.0f * reservedRight);

    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * config.titleScale);
    const F32 titleWidth = ImGui::CalcTextSize(config.title.c_str()).x;
    std::string displayTitle = config.title;
    F32 titleZoneWidth = availWidth;
    if (titleWidth > fullCenterSafeWidth) {
        if (titleWidth > titleAvailWidth) {
            const F32 ellipsisWidth = ImGui::CalcTextSize("...").x;
            const F32 budget = ImMax(0.0f, titleAvailWidth - ellipsisWidth);
            std::size_t cut = 0;
            while (cut < displayTitle.size() &&
                   ImGui::CalcTextSize(displayTitle.c_str(),
                                       displayTitle.c_str() + cut + 1).x <= budget) {
                ++cut;
            }
            displayTitle.resize(cut);
            displayTitle += "...";
        }
        titleZoneWidth = titleAvailWidth;
    }
    const F32 displayWidth = ImGui::CalcTextSize(displayTitle.c_str()).x;
    const F32 titleOffsetX = ImMax(0.0f, (titleZoneWidth - displayWidth) * 0.5f);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + titleOffsetX);
    ImGui::TextUnformatted(displayTitle.c_str(),
                           displayTitle.c_str() + displayTitle.size());
    ImGui::PopFont();

    const ImVec2 titleMin = ImGui::GetItemRectMin();
    const ImVec2 titleMax = ImGui::GetItemRectMax();

    ImVec2 chevronIconMin(0.0f, 0.0f);
    ImVec2 chevronIconMax(0.0f, 0.0f);
    if (hasChevron) {
        chevronIconMin.x = titleStartScreen.x + availWidth - chevronSize.x;
        chevronIconMin.y =
            titleMin.y +
            ImMax(0.0f, (titleMax.y - titleMin.y - chevronSize.y) * 0.5f);
        chevronIconMax = ImVec2(chevronIconMin.x + chevronSize.x,
                                chevronIconMin.y + chevronSize.y);
    }

    ImVec2 diagnosticIconMin(0.0f, 0.0f);
    ImVec2 diagnosticIconMax(0.0f, 0.0f);
    if (hasDiagnostic) {
        F32 skullX = titleStartScreen.x + availWidth - skullSize.x;
        if (hasChevron) {
            skullX = chevronIconMin.x - gap - skullSize.x;
        }
        diagnosticIconMin.x = skullX;
        diagnosticIconMin.y =
            titleMin.y +
            ImMax(0.0f, (titleMax.y - titleMin.y - skullSize.y) * 0.5f);
        diagnosticIconMax = ImVec2(diagnosticIconMin.x + skullSize.x,
                                   diagnosticIconMin.y + skullSize.y);
    }

    ImNodes::EndNodeTitleBar();

    if (hasChevron) {
        const bool hoveringChevron =
            ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
            ImGui::IsMouseHoveringRect(chevronIconMin, chevronIconMax);
        if (hoveringChevron && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            impl->chevronArmed = true;
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            if (impl->chevronArmed && hoveringChevron) {
                const ImVec2 drag = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                if (std::fabs(drag.x) < Scale(ctx, 3.0f) &&
                    std::fabs(drag.y) < Scale(ctx, 3.0f)) {
                    config.onToggleConfigCollapse();
                }
            }
            impl->chevronArmed = false;
        }
        if (hoveringChevron) {
            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
        }
        const char* chevronIcon = config.configCollapsed
            ? ICON_FA_CHEVRON_RIGHT
            : ICON_FA_CHEVRON_DOWN;
        const std::string chevronColorKey = hoveringChevron
            ? "text_primary"
            : "text_secondary";
        const ImU32 chevronColor =
            ImGui::ColorConvertFloat4ToU32(Private::ImColor(ctx, chevronColorKey));
        ImGui::GetWindowDrawList()->AddText(chevronIconMin,
                                            chevronColor,
                                            chevronIcon);
    }

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
