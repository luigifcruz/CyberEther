#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_PLUGIN_PROMPT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_PLUGIN_PROMPT_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "jetstream/types.hh"

#include <string>
#include <vector>

namespace Jetstream {

struct PluginPromptView {
    static constexpr F32 Width = 480.0f;
    static constexpr F32 ButtonHeight = 34.0f;

    struct Config {
        std::string title;
        std::string description;
        std::vector<std::string> columns;
        std::vector<std::vector<std::string>> rows;
        std::string cancel;
        std::string action;
        bool busy = false;
        Sakura::Callback<void()> onAction;
        Sakura::Callback<void()> onCancel;
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
    Sakura::Table table;
    Sakura::Spacer footerSpacer;
    Sakura::PillButton cancel;
    Sakura::PillButton action;
    Sakura::Stack footer;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_PLUGIN_PROMPT_HH
