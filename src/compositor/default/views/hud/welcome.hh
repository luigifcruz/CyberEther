#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_WELCOME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_WELCOME_HH

#include "../components/assistant_composer.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct WelcomeHudTitle : public Sakura::Stack {
    struct Config {
        std::string title;
        std::string slogan;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        title.update({
            .str = config.title,
            .font = Text::Font::Display,
            .align = Text::Align::Center,
            .colorKey = "cyber_blue",
            .scale = 8.5f,
        });
        slogan.update({
            .str = config.slogan,
            .font = Text::Font::Display,
            .tone = Text::Tone::Secondary,
            .align = Text::Align::Center,
            .scale = 2.3f,
        });
        Stack::update({
            .items = {
                {.child = &title},
                {.child = &slogan},
            },
        });
    }

 private:
    Sakura::Text title;
    Sakura::Text slogan;
};

struct WelcomeHudCell : public Sakura::Div {
    struct Config {
        std::string icon;
        std::string title;
        std::string iconColorKey;
        std::function<void()> onClick;
    };

    void update(Config config) {
        using Text = Sakura::Text;

        this->config = std::move(config);
        icon.update({
            .str = this->config.icon,
            .align = Text::Align::Center,
            .colorKey = this->config.iconColorKey,
            .scale = 2.0f,
            .verticalOffset = 13.0f,
        });
        title.update({
            .str = this->config.title,
            .font = Text::Font::Bold,
            .align = Text::Align::Center,
            .scale = 1.15f,
            .verticalOffset = 4.0f,
        });
        content.update({
            .items = {
                {.child = &icon},
                {.child = &title},
            },
        });
        Div::update({
            .content = &content,
            .size = {Sakura::Fill, 118.0f},
            .padding = 16.0f,
            .rounding = 16.0f,
            .border = true,
            .chrome = true,
            .colorKey = "welcome_card_bg",
            .hoveredColorKey = "welcome_card_hovered",
            .onClick = [this]() {
                if (this->config.onClick) {
                    this->config.onClick();
                }
            },
        });
    }

 private:
    Config config;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Stack content;
};

struct WelcomeHudView {
    struct Config {
        std::string version;
        bool chatOpen = false;
        bool assistantEnabled = true;
        U64 assistantClearRequest = 0;
        std::function<void(const std::string&)> onAssistantSubmit;
        std::function<void()> onAssistantVoice;
        std::function<void()> onNewFlowgraph;
        std::function<void()> onOpenFlowgraph;
        std::function<void()> onOpenExamples;
        std::function<void()> onOpenWebsite;
        std::function<void()> onOpenDocs;
        std::function<void()> onOpenSettings;
        std::function<void()> onOpenAbout;
    };

    void update(Config config) {
        this->config = std::move(config);
        title.update({
            .title = "CyberEther",
            .slogan = "GPU-ACCELERATED SIGNAL PROCESSING",
        });
        welcomeSpacing.update({.lines = 3});
        assistant.update({
            .id = "WelcomeAssistant",
            .style = AssistantComposerStyle({
                .backgroundColorKey = "card",
                .compact = true,
                .margin = {.left = 4.0f, .right = 4.0f, .bottom = 6.0f},
            }),
            .onSubmit = this->config.onAssistantSubmit,
            .onVoice = this->config.onAssistantVoice,
            .clearRequest = this->config.assistantClearRequest,
        });
        assistantSpacing.update({});
        buttonSpacing.update({});
        versionSpacing.update({});
        newCell.update({
            .icon = ICON_LC_FILE_PLUS_2,
            .title = "New Flowgraph",
            .iconColorKey = "welcome_icon_new",
            .onClick = [this]() {
                if (this->config.onNewFlowgraph) {
                    this->config.onNewFlowgraph();
                }
            },
        });
        openCell.update({
            .icon = ICON_LC_FOLDER_OPEN,
            .title = "Open Flowgraph",
            .iconColorKey = "welcome_icon_open",
            .onClick = [this]() {
                if (this->config.onOpenFlowgraph) {
                    this->config.onOpenFlowgraph();
                }
            },
        });
        examplesCell.update({
            .icon = ICON_LC_FLASK_CONICAL,
            .title = "Examples",
            .iconColorKey = "welcome_icon_examples",
            .onClick = [this]() {
                if (this->config.onOpenExamples) {
                    this->config.onOpenExamples();
                }
            },
        });
        websiteButton.update({
            .str = ICON_LC_GLOBE " Website",
            .size = {Sakura::Fill, 34.0f},
            .chrome = true,
            .colorKey = "welcome_card_bg",
            .hoveredColorKey = "welcome_card_hovered",
            .activeColorKey = "welcome_card_bg",
            .onClick = this->config.onOpenWebsite,
        });
        docsButton.update({
            .str = ICON_LC_BOOK " Docs",
            .size = {Sakura::Fill, 34.0f},
            .chrome = true,
            .colorKey = "welcome_card_bg",
            .hoveredColorKey = "welcome_card_hovered",
            .activeColorKey = "welcome_card_bg",
            .onClick = this->config.onOpenDocs,
        });
        settingsButton.update({
            .str = ICON_LC_SETTINGS " Settings",
            .size = {Sakura::Fill, 34.0f},
            .chrome = true,
            .colorKey = "welcome_card_bg",
            .hoveredColorKey = "welcome_card_hovered",
            .activeColorKey = "welcome_card_bg",
            .onClick = this->config.onOpenSettings,
        });
        aboutButton.update({
            .str = ICON_LC_INFO " About",
            .size = {Sakura::Fill, 34.0f},
            .chrome = true,
            .colorKey = "welcome_card_bg",
            .hoveredColorKey = "welcome_card_hovered",
            .activeColorKey = "welcome_card_bg",
            .onClick = this->config.onOpenAbout,
        });
        version.update({
            .str = this->config.version,
            .tone = Sakura::Text::Tone::Disabled,
            .align = Sakura::Text::Align::Center,
        });
        actions.update({
            .columns = 3,
            .size = {Sakura::Fill, 126.0f},
            .cellPadding = {4.0f, 2.0f},
            .items = {&newCell, &openCell, &examplesCell},
        });
        links.update({
            .columns = 4,
            .size = {Sakura::Fill, 42.0f},
            .cellPadding = {4.0f, 2.0f},
            .items = {&websiteButton, &docsButton, &settingsButton, &aboutButton},
        });
        std::vector<Sakura::Stack::Item> items = {
            {.child = &title},
            {.child = &welcomeSpacing},
        };
        if (this->config.assistantEnabled && !this->config.chatOpen) {
            items.push_back({.child = &assistant});
            items.push_back({.child = &assistantSpacing});
        }
        items.push_back({.child = &actions});
        items.push_back({.child = &buttonSpacing});
        items.push_back({.child = &links});
        items.push_back({.child = &versionSpacing});
        items.push_back({.child = &version});
        stack.update({.items = std::move(items)});
        panel.update({
            .id = "Welcome",
            .content = &stack,
            .transparent = true,
        });
    }

    Sakura::Component& component() {
        return panel;
    }

    Sakura::HudLayer::Hud hud() {
        return {
            .content = &panel,
            .anchor = Sakura::Anchor::Center,
            .margin = 0.0f,
            .padding = {2.0f, 0.0f},
            .borderSize = 0.0f,
            .backgroundColorKey = "transparent",
            .borderColorKey = "transparent",
            .size = Extent2D<F32>{560.0f, 0.0f},
        };
    }

 private:
    Config config;
    WelcomeHudTitle title;
    Sakura::Spacer welcomeSpacing;
    Sakura::ChatComposer assistant;
    Sakura::Spacer assistantSpacing;
    Sakura::Spacer buttonSpacing;
    Sakura::Spacer versionSpacing;
    WelcomeHudCell newCell;
    WelcomeHudCell openCell;
    WelcomeHudCell examplesCell;
    Sakura::Button websiteButton;
    Sakura::Button docsButton;
    Sakura::Button settingsButton;
    Sakura::Button aboutButton;
    Sakura::Text version;
    Sakura::Grid actions;
    Sakura::Grid links;
    Sakura::Stack stack;
    Sakura::Panel panel;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_WELCOME_HH
