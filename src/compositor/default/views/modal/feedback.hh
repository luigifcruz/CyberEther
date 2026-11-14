#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_FEEDBACK_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/types.hh"

#include <string>

namespace Jetstream {

struct FeedbackView {
    static constexpr F32 EditorHeight = 180.0f;
    static constexpr F32 ButtonHeight = 34.0f;
    static constexpr F32 CountScale = Sakura::Typography::Scale::Small;

    struct Config {
        std::string text;
        U64 minLength = 10;
        U64 maxLength = 4000;
        enum class Status { Idle, Submitting };
        Status status = Status::Idle;
        Sakura::Callback<void(const std::string&)> onTextChange;
        Sakura::Callback<void()> onSubmit;
        Sakura::Callback<void()> onClose;
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
    Sakura::CodeEditor editor;
    Sakura::Div editorCard;
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
