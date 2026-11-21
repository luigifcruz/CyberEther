#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CLOSE_PROMPT_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CLOSE_PROMPT_HH

#include "../components/card_button.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <optional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphClosePromptView {
    static constexpr F32 Width = 380.0f;
    static constexpr F32 ButtonHeight = 34.0f;

    struct Config {
        std::string flowgraph;
        Sakura::Callback<void()> onSave;
        Sakura::Callback<void()> onDontSave;
        Sakura::Callback<void()> onCancel;
    };

    void update(Config config);
    void close();
    Sakura::ModalLayer::Config layer(std::optional<Rect> anchor);

 private:
    void refresh(bool open);

    Config config;
    bool open = false;
    Sakura::Text title;
    Sakura::Text message;
    Sakura::Spacer gap;
    CardButton cancel;
    CardButton discard;
    CardButton save;
    Sakura::Grid buttons;
    Sakura::Stack stack;
    Sakura::Div frame;
    Sakura::Shortcuts keys;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_CLOSE_PROMPT_HH
