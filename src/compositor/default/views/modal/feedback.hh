#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/types.hh"

#include <memory>
#include <string>

namespace Jetstream {

struct FeedbackView {
    static constexpr F32 EditorHeight = 180.0f;
    static constexpr F32 ReportEditorHeight = 120.0f;
    static constexpr F32 DebugEditorHeight = 180.0f;
    static constexpr F32 AreaInset = 6.0f;
    static constexpr F32 AreaPadding = 8.0f;
    static constexpr F32 AreaVerticalPadding = 6.0f;
    static constexpr F32 AreaBorder = 1.5f;
    static constexpr F32 AreaShadowSoftness = 4.0f;
    static constexpr F32 AreaShadowDrop = 1.0f;
    static constexpr const char* AreaFill = "pill_field_bg";
    static constexpr F32 ButtonHeight = 34.0f;
    static constexpr F32 CountScale = Sakura::Typography::Scale::Small;

    struct Config {
        U64 session = 0;
        std::string text;
        std::string debugInfo;
        bool blockReport = false;
        U64 minLength = 10;
        U64 maxLength = 4000;
        enum class Status { Idle, Submitting };
        Status status = Status::Idle;
        Sakura::Callback<void(const std::string&)> onTextChange;
        Sakura::Callback<void(const std::string&)> onDebugInfoChange;
        Sakura::Callback<void()> onSubmit;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    Config config;
    U64 editorSession = 0;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Stack titleRow;
    Sakura::Text description;
    Sakura::Stack header;
    std::unique_ptr<Sakura::CodeEditor> editor;
    Sakura::Text editorHint;
    Sakura::Overlay editorWrite;
    Sakura::Div editorCard;
    Sakura::Text debugLabel;
    std::unique_ptr<Sakura::CodeEditor> debugEditor;
    Sakura::Div debugCard;
    Sakura::Stack debugSection;
    Sakura::Text count;
    Sakura::Spacer footerSpacer;
    Sakura::PillButton cancel;
    Sakura::PillButton send;
    Sakura::Stack footer;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH
