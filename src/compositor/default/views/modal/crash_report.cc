#include "crash_report.hh"

namespace Jetstream {

void CrashReportView::update(Config config) {
    this->config = std::move(config);

    using Text = Sakura::Text;

    icon.update({
        .str = ICON_LC_BUG,
        .colorKey = "text_primary",
        .scale = Sakura::Typography::Scale::Heading,
    });
    title.update({
        .str = "CyberEther Quit Unexpectedly",
        .font = Text::Font::Display,
        .scale = Sakura::Typography::Scale::Headline,
    });
    titleRow.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &icon},
            {.child = &title},
        },
    });
    description.update({
        .str = this->config.count > 1
                   ? jst::fmt::format("{} crash reports were saved from previous sessions. "
                                      "Sending them helps find and fix the problem. "
                                      "They are anonymous and the most recent one is shown in full below.",
                                      this->config.count)
                   : "A crash report was saved from a previous session. "
                     "Sending it helps find and fix the problem. "
                     "It is anonymous and shown in full below.",
        .font = Text::Font::Sans,
        .tone = Text::Tone::Secondary,
        .wrapped = true,
        .scale = Sakura::Typography::Scale::Lead,
    });
    header.update({
        .spacing = Sakura::Spacing::XSmall,
        .items = {
            {.child = &titleRow},
            {.child = &description},
        },
    });

    report.update({
        .value = this->config.report,
        .height = ReportHeight - 2.0f * ReportInset,
        .padding = ReportPadding,
        .backgroundColorKey = ReportFill,
        .textColorKey = "text_primary",
    });
    reportCard.update({
        .content = &report,
        .size = {Sakura::Fill, ReportHeight},
        .padding = ReportInset,
        .rounding = Sakura::Rounding::Card,
        .border = true,
        .borderWidth = ReportBorder,
        .chrome = true,
        .shadow = true,
        .shadowSoftness = ReportShadowSoftness,
        .shadowDrop = ReportShadowDrop,
        .colorKey = ReportFill,
        .hoveredColorKey = ReportFill,
    });

    dismiss.update({
        .label = "Don't Send",
        .hoverColorKey = "chrome_button_hover",
        .height = ButtonHeight,
        .onClick = [this]() {
            if (this->config.onDismiss) {
                this->config.onDismiss();
            }
        },
    });
    send.update({
        .icon = ICON_LC_SEND,
        .label = "Send Report",
        .colorKey = "action_btn_text",
        .backgroundColorKey = "action_btn",
        .hoverColorKey = "action_btn_hovered",
        .raised = true,
        .height = ButtonHeight,
        .onClick = [this]() {
            if (this->config.onSubmit) {
                this->config.onSubmit();
            }
        },
    });
    footerSpacer.update({});
    footer.update({
        .axis = Sakura::Stack::Axis::Horizontal,
        .spacing = Sakura::Spacing::Small,
        .items = {
            {.child = &footerSpacer, .grow = 1.0f},
            {.child = &dismiss},
            {.child = &send},
        },
    });

    stack.update({
        .spacing = Sakura::Spacing::Large,
        .items = {
            {.child = &header},
            {.child = &reportCard},
            {.child = &footer},
        },
    });
    panel.update({
        .id = "CrashReportModal",
        .content = &stack,
        .backgroundColorKey = "popup_bg",
    });
}

Sakura::Component& CrashReportView::component() {
    return panel;
}

}  // namespace Jetstream
