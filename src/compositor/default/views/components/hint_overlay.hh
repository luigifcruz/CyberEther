#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_HINT_OVERLAY_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_HINT_OVERLAY_HH

#include "card_button.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct HintOverlay {
    static constexpr F32 BadgeSize = 26.0f;
    static constexpr F32 BadgeText = Sakura::Typography::FontSize * Sakura::Typography::Scale::Small;
    static constexpr F32 ButtonHeight = 40.0f;

    struct Action {
        std::string label;
        CardButton::Tone tone = CardButton::Tone::Card;
        Sakura::Callback<void()> onClick;
    };

    struct Config {
        std::string icon = ICON_LC_LIGHTBULB;
        std::string title;
        std::string subtitle;
        std::vector<std::string> steps;
        std::vector<Action> actions;
        F32 width = 420.0f;
    };

    void update(Config config);
    Sakura::Component& component();
    Extent2D<F32> size() const;

 private:
    struct Step {
        Sakura::Text number;
        Sakura::Div badge;
        Sakura::Text text;
        Sakura::Stack row;
    };

    void updateSteps();
    void updateActions();

    Config config;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Stack heading;
    Sakura::Text subtitle;
    Sakura::Spacer headerSpacing;
    std::vector<std::unique_ptr<Step>> stepRows;
    Sakura::Stack steps;
    Sakura::Spacer stepsSpacing;
    std::vector<std::unique_ptr<CardButton>> actionButtons;
    Sakura::Stack stack;
    Sakura::Div card;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_HINT_OVERLAY_HH
