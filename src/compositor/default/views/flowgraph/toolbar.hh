#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_TOOLBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_TOOLBAR_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include <functional>
#include <string>
#include <utility>

namespace Jetstream {

struct FlowgraphToolbar {
    struct Config {
        std::string id;
        bool dependencyReviewAvailable = false;
        std::string dependencyReviewMessage;
        std::function<void()> onSave;
        std::function<void()> onClose;
        std::function<void()> onAddBlock;
        std::function<void()> onCreateStack;
        std::function<void()> onSendFeedback;
        std::function<void()> onReviewDependencies;
    };

    void update(Config config) {
        this->config = std::move(config);
        const auto toolbarSize = this->config.dependencyReviewAvailable
                                     ? expandedToolbarSize
                                     : collapsedToolbarSize;
        overlay.update({
            .id = this->config.id + ":overlay",
            .size = toolbarSize,
            .anchor = Sakura::Overlay::Anchor::TopCenter,
            .offset = {0.0f, 12.0f},
            .inputs = true,
        });
        card.update({
            .id = this->config.id + ":card",
            .size = toolbarSize,
            .padding = 6.0f,
            .rounding = 12.0f,
            .border = true,
            .scrollbar = false,
            .mouseScroll = false,
        });
        layout.update({
            .id = this->config.id + ":layout",
            .spacing = 0.0f,
        });
        saveButton.update({
            .id = this->config.id + ":save",
            .str = ICON_FA_FLOPPY_DISK " Save",
            .size = {82.0f, 34.0f},
            .onClick = this->config.onSave,
        });
        closeButton.update({
            .id = this->config.id + ":close",
            .str = ICON_FA_CIRCLE_XMARK " Close",
            .size = {86.0f, 34.0f},
            .onClick = this->config.onClose,
        });
        addBlockButton.update({
            .id = this->config.id + ":add-block",
            .str = ICON_FA_CUBE " Blocks",
            .size = {90.0f, 34.0f},
            .onClick = this->config.onAddBlock,
        });
        createStackButton.update({
            .id = this->config.id + ":create-stack",
            .str = ICON_FA_LAYER_GROUP " Stack",
            .size = {90.0f, 34.0f},
            .onClick = this->config.onCreateStack,
        });
        feedbackButton.update({
            .id = this->config.id + ":feedback",
            .str = ICON_FA_COMMENT_DOTS " Feedback",
            .size = {100.0f, 34.0f},
            .onClick = this->config.onSendFeedback,
        });
        banner.update({
            .id = this->config.id + ":banner",
            .size = {0.0f, 32.0f},
            .padding = 5.0f,
            .rounding = 8.0f,
            .border = true,
            .scrollbar = false,
            .mouseScroll = false,
            .colorKey = "banner_info_bg",
            .borderColorKey = "banner_info_border",
        });
        bannerLayout.update({
            .id = this->config.id + ":banner-layout",
            .spacing = 0.0f,
        });
        bannerIndent.update({
            .id = this->config.id + ":banner-indent",
        });
        bannerTextContainer.update({
            .id = this->config.id + ":banner-text-container",
            .size = {382.0f, 22.0f},
            .border = false,
            .scrollbar = false,
            .mouseScroll = false,
            .inputs = false,
        });
        bannerText.update({
            .id = this->config.id + ":banner-text",
            .str = this->config.dependencyReviewMessage,
            .colorKey = "banner_info_text",
            .scale = 1.0f,
            .verticalOffset = 3.0f,
        });
        reviewButtonContainer.update({
            .id = this->config.id + ":review-container",
            .size = {72.0f, 22.0f},
            .padding = 1.0f,
            .border = false,
            .scrollbar = false,
            .mouseScroll = false,
            .colorKey = "transparent",
        });
        reviewButton.update({
            .id = this->config.id + ":review",
            .str = "Review",
            .size = {70.0f, 20.0f},
            .variant = Sakura::Button::Variant::Action,
            .rounding = 5.0f,
            .onClick = this->config.onReviewDependencies,
        });
    }

    void render(const Sakura::Context& ctx) {
        overlay.render(ctx, [this](const Sakura::Context& ctx) {
            card.render(ctx, [this](const Sakura::Context& ctx) {
                layout.render(ctx, {
                    [this](const Sakura::Context& ctx) { saveButton.render(ctx); },
                    [this](const Sakura::Context& ctx) { closeButton.render(ctx); },
                    [this](const Sakura::Context& ctx) { addBlockButton.render(ctx); },
                    [this](const Sakura::Context& ctx) { createStackButton.render(ctx); },
                    [this](const Sakura::Context& ctx) { feedbackButton.render(ctx); },
                });
                if (config.dependencyReviewAvailable) {
                    banner.render(ctx, [this](const Sakura::Context& ctx) {
                        bannerLayout.render(ctx, {
                            [this](const Sakura::Context& ctx) { bannerIndent.render(ctx); },
                            [this](const Sakura::Context& ctx) {
                                bannerTextContainer.render(ctx, [this](const Sakura::Context& ctx) {
                                    bannerText.render(ctx);
                                });
                            },
                            [this](const Sakura::Context& ctx) {
                                reviewButtonContainer.render(ctx, [this](const Sakura::Context& ctx) {
                                    reviewButton.render(ctx);
                                });
                            },
                        });
                    });
                }
            });
        });
    }

 private:
    static constexpr Extent2D<F32> collapsedToolbarSize = {492.0f, 46.0f};
    static constexpr Extent2D<F32> expandedToolbarSize = {492.0f, 86.0f};

    Config config;
    Sakura::Overlay overlay;
    Sakura::Div card;
    Sakura::HStack layout;
    Sakura::Button saveButton;
    Sakura::Button closeButton;
    Sakura::Button addBlockButton;
    Sakura::Button createStackButton;
    Sakura::Button feedbackButton;
    Sakura::Div banner;
    Sakura::HStack bannerLayout;
    Sakura::Spacing bannerIndent;
    Sakura::Div bannerTextContainer;
    Sakura::Text bannerText;
    Sakura::Div reviewButtonContainer;
    Sakura::Button reviewButton;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_TOOLBAR_HH
