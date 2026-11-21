#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_PLACEHOLDER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_PLACEHOLDER_HH

#include "../components/card_button.hh"

#include "jetstream/render/sakura/base.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct AgentChatPlaceholder {
    static constexpr F32 SuggestionHeight = 44.0f;
    static constexpr F32 SetupHeight = 40.0f;
    static constexpr F32 TitleScale = Sakura::Typography::Scale::Headline;
    static constexpr F32 RowSpacing = Sakura::Spacing::Small;
    static constexpr F32 SectionSpacing = Sakura::Spacing::XLarge;

    struct Suggestion {
        std::string icon;
        std::string label;
        Sakura::Callback<void()> onClick;
    };

    struct Config {
        std::string title;
        std::string subtitle;
        std::vector<Suggestion> suggestions;
        std::string setupLabel;
        Sakura::Callback<void()> onSetup;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    Sakura::Text title;
    Sakura::Text subtitle;
    Sakura::Stack header;
    std::vector<std::unique_ptr<Sakura::Button>> buttons;
    CardButton setup;
    Sakura::Stack actions;
    Sakura::Stack stack;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_AGENT_PLACEHOLDER_HH
