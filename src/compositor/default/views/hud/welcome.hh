#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_WELCOME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_WELCOME_HH

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct WelcomeHudTitle : public Sakura::Component {
    struct Config {
        std::string title;
        std::string slogan;
    };

    void update(Config config) {
        this->config = std::move(config);
        title.update({
            .id = "WelcomeTitle",
            .str = this->config.title,
            .font = Sakura::Text::Font::Bold,
            .align = Sakura::Text::Align::Center,
            .colorKey = "cyber_blue",
            .scale = 6.0f,
        });
        slogan.update({
            .id = "WelcomeSlogan",
            .str = this->config.slogan,
            .tone = Sakura::Text::Tone::Secondary,
            .align = Sakura::Text::Align::Center,
            .scale = 1.3f,
        });
    }

    void render(const Sakura::Context& ctx) const {
        title.render(ctx);
        slogan.render(ctx);
    }

 private:
    Config config;
    Sakura::Text title;
    Sakura::Text slogan;
};

struct WelcomeHudCell : public Sakura::Component {
    struct Config {
        std::string id;
        std::string icon;
        std::string title;
        std::string subtitle;
        std::string iconColorKey;
        std::function<void()> onClick;
    };

    void update(Config config) {
        this->config = std::move(config);
        card.update({
            .id = this->config.id,
            .size = {176.0f, 118.0f},
            .padding = 22.0f,
            .rounding = 16.0f,
            .border = true,
            .scrollbar = false,
            .mouseScroll = false,
            .colorKey = "card",
            .hoveredColorKey = "header_hovered",
            .borderColorKey = "border",
            .onClick = [this]() {
                if (this->config.onClick) {
                    this->config.onClick();
                }
            },
        });
        content.update({
            .id = this->config.id + "Content",
        });
        icon.update({
            .id = this->config.id + "Icon",
            .str = this->config.icon,
            .align = Sakura::Text::Align::Center,
            .colorKey = this->config.iconColorKey,
            .scale = 2.0f,
        });
        title.update({
            .id = this->config.id + "Title",
            .str = this->config.title,
            .font = Sakura::Text::Font::Bold,
            .align = Sakura::Text::Align::Center,
        });
        subtitle.update({
            .id = this->config.id + "Subtitle",
            .str = this->config.subtitle,
            .tone = Sakura::Text::Tone::Secondary,
            .align = Sakura::Text::Align::Center,
        });
    }

    void render(const Sakura::Context& ctx) const {
        card.render(ctx, [this](const Sakura::Context& ctx) {
            content.render(ctx, {
                [this](const Sakura::Context& ctx) { icon.render(ctx); },
                [this](const Sakura::Context& ctx) { title.render(ctx); },
                [this](const Sakura::Context& ctx) { subtitle.render(ctx); },
            });
        });
    }

 private:
    Config config;
    Sakura::Div card;
    Sakura::VStack content;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Text subtitle;
};

struct WelcomeHudView : public Sakura::Component {
    struct Config {
        bool visible = false;
        std::string version;
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
        hud.update({
            .id = "Welcome",
            .anchor = Sakura::Hud::Anchor::Center,
            .padding = 0.0f,
            .size = Extent2D<F32>{560.0f, 0.0f},
            .windowPadding = Extent2D<F32>{2.0f, 0.0f},
            .borderSize = 0.0f,
            .backgroundAlpha = 0.0f,
            .borderAlpha = 0.0f,
        });
        title.update({
            .title = "CyberEther",
            .slogan = "GPU-accelerated Signal Processing",
        });
        actions.update({
            .id = "WelcomeActions",
            .columns = 3,
            .size = {560.0f, 126.0f},
            .cellPadding = {4.0f, 2.0f},
        });
        links.update({
            .id = "WelcomeLinks",
            .columns = 4,
            .size = {560.0f, 42.0f},
            .cellPadding = {4.0f, 2.0f},
        });
        spacing.update({
            .id = "WelcomeSpacing",
            .lines = 3,
        });
        newCell.update({
            .id = "WelcomeNewCard",
            .icon = ICON_FA_FILE_CIRCLE_PLUS,
            .title = "New Flowgraph",
            .subtitle = "Start fresh",
            .iconColorKey = "welcome_icon_new",
            .onClick = [this]() {
                if (this->config.onNewFlowgraph) {
                    this->config.onNewFlowgraph();
                }
            },
        });
        openCell.update({
            .id = "WelcomeOpenCard",
            .icon = ICON_FA_FOLDER_OPEN,
            .title = "Open File",
            .subtitle = "Load existing",
            .iconColorKey = "welcome_icon_open",
            .onClick = [this]() {
                if (this->config.onOpenFlowgraph) {
                    this->config.onOpenFlowgraph();
                }
            },
        });
        examplesCell.update({
            .id = "WelcomeExamplesCard",
            .icon = ICON_FA_FLASK,
            .title = "Examples",
            .subtitle = "Quick start",
            .iconColorKey = "welcome_icon_examples",
            .onClick = [this]() {
                if (this->config.onOpenExamples) {
                    this->config.onOpenExamples();
                }
            },
        });
        websiteButton.update({
            .id = "WelcomeWebsite",
            .str = ICON_FA_GLOBE " Website",
            .size = {-FLT_MIN, 34.0f},
            .onClick = this->config.onOpenWebsite,
        });
        docsButton.update({
            .id = "WelcomeDocs",
            .str = ICON_FA_BOOK " Docs",
            .size = {-FLT_MIN, 34.0f},
            .onClick = this->config.onOpenDocs,
        });
        settingsButton.update({
            .id = "WelcomeSettings",
            .str = ICON_FA_GEAR " Settings",
            .size = {-FLT_MIN, 34.0f},
            .onClick = this->config.onOpenSettings,
        });
        aboutButton.update({
            .id = "WelcomeAbout",
            .str = ICON_FA_CIRCLE_INFO " About",
            .size = {-FLT_MIN, 34.0f},
            .onClick = this->config.onOpenAbout,
        });
        version.update({
            .id = "WelcomeVersion",
            .str = this->config.version,
            .tone = Sakura::Text::Tone::Disabled,
            .align = Sakura::Text::Align::Center,
        });
    }

    void render(const Sakura::Context& ctx) const {
        if (!config.visible) {
            return;
        }

        hud.render(ctx, [this](const Sakura::Context& ctx) {
            title.render(ctx);
            spacing.render(ctx);
            actions.render(ctx, {
                [this](const Sakura::Context& ctx) { newCell.render(ctx); },
                [this](const Sakura::Context& ctx) { openCell.render(ctx); },
                [this](const Sakura::Context& ctx) { examplesCell.render(ctx); },
            });
            spacing.render(ctx);
            links.render(ctx, {
                [this](const Sakura::Context& ctx) { websiteButton.render(ctx); },
                [this](const Sakura::Context& ctx) { docsButton.render(ctx); },
                [this](const Sakura::Context& ctx) { settingsButton.render(ctx); },
                [this](const Sakura::Context& ctx) { aboutButton.render(ctx); },
            });
            spacing.render(ctx);
            version.render(ctx);
        });
    }

 private:
    Config config;
    Sakura::Hud hud;
    WelcomeHudTitle title;
    Sakura::Grid actions;
    Sakura::Grid links;
    Sakura::Spacing spacing;
    WelcomeHudCell newCell;
    WelcomeHudCell openCell;
    WelcomeHudCell examplesCell;
    Sakura::Button websiteButton;
    Sakura::Button docsButton;
    Sakura::Button settingsButton;
    Sakura::Button aboutButton;
    Sakura::Text version;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_HUD_WELCOME_HH
