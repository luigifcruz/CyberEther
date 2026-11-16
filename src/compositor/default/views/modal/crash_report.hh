#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CRASH_REPORT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CRASH_REPORT_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/types.hh"

#include <string>

namespace Jetstream {

struct CrashReportView {
    static constexpr F32 Width = 560.0f;
    static constexpr F32 ReportHeight = 200.0f;
    static constexpr F32 ReportInset = 6.0f;
    static constexpr F32 ReportPadding = 8.0f;
    static constexpr F32 ReportBorder = 1.5f;
    static constexpr F32 ReportShadowSoftness = 4.0f;
    static constexpr F32 ReportShadowDrop = 1.0f;
    static constexpr const char* ReportFill = "pill_field_bg";
    static constexpr F32 ButtonHeight = 34.0f;

    struct Config {
        std::string report;
        U64 count = 0;
        Sakura::Callback<void()> onSubmit;
        Sakura::Callback<void()> onDismiss;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Stack titleRow;
    Sakura::Text description;
    Sakura::Stack header;
    Sakura::LogView report;
    Sakura::Div reportCard;
    Sakura::Spacer footerSpacer;
    Sakura::PillButton dismiss;
    Sakura::PillButton send;
    Sakura::Stack footer;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_CRASH_REPORT_HH
