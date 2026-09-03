#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_TOOLBAR_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_TOOLBAR_HH

#include "../components/callout.hh"

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
        Callout::Tone dependencyReviewTone = Callout::Tone::Info;
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
        const std::string bannerKey = "banner_" + bannerToneKey();
        banner.update({
            .id = this->config.id + ":banner",
            .size = {0.0f, bannerHeight},
            .padding = bannerPadding,
            .rounding = 8.0f,
            .border = true,
            .scrollbar = false,
            .mouseScroll = false,
            .colorKey = bannerKey + "_bg",
            .borderColorKey = bannerKey + "_border",
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
            .size = {382.0f, bannerRowHeight},
            .border = false,
            .scrollbar = false,
            .mouseScroll = false,
            .inputs = false,
        });
        bannerText.update({
            .id = this->config.id + ":banner-text",
            .str = std::string(ICON_FA_BOX) + " " + this->config.dependencyReviewMessage,
            .colorKey = bannerKey + "_text",
            .scale = 1.0f,
            .verticalOffset = bannerTextOffset,
        });
        reviewButtonContainer.update({
            .id = this->config.id + ":review-container",
            .size = {72.0f, bannerRowHeight},
            .padding = reviewButtonPadding,
            .border = false,
            .scrollbar = false,
            .mouseScroll = false,
            .colorKey = "transparent",
        });
        reviewButton.update({
            .id = this->config.id + ":review",
            .str = "Review",
            .size = {70.0f, bannerRowHeight - 2.0f * reviewButtonPadding},
            .variant = reviewButtonVariant(),
            .colorKey = reviewButtonKey("warning_btn", "button"),
            .hoveredColorKey = reviewButtonKey("warning_btn_hovered", "button_hovered"),
            .activeColorKey = reviewButtonKey("warning_btn_active", "button_active"),
            .borderColorKey = reviewButtonKey("warning_btn_outline", "button_outline"),
            .textColorKey = reviewButtonKey("warning_btn_text", "button_text"),
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
    std::string bannerToneKey() const {
        switch (config.dependencyReviewTone) {
            case Callout::Tone::Error:
                return "error";
            case Callout::Tone::Warning:
                return "warning";
            case Callout::Tone::Success:
                return "success";
            case Callout::Tone::Info:
                break;
        }
        return "info";
    }

    Sakura::Button::Variant reviewButtonVariant() const {
        switch (config.dependencyReviewTone) {
            case Callout::Tone::Error:
                return Sakura::Button::Variant::Destructive;
            case Callout::Tone::Warning:
                return Sakura::Button::Variant::Default;
            default:
                return Sakura::Button::Variant::Action;
        }
    }

    std::string reviewButtonKey(const char* warningKey, const char* defaultKey) const {
        return config.dependencyReviewTone == Callout::Tone::Warning ? warningKey : defaultKey;
    }

    static constexpr F32 bannerPadding = 5.0f;
    static constexpr F32 bannerRowHeight = 26.0f;
    static constexpr F32 bannerHeight = bannerRowHeight + 2.0f * bannerPadding;
    static constexpr F32 bannerTextOffset = 5.0f;
    static constexpr F32 reviewButtonPadding = 1.0f;
    static constexpr Extent2D<F32> collapsedToolbarSize = {492.0f, 46.0f};
    static constexpr Extent2D<F32> expandedToolbarSize = {collapsedToolbarSize.x,
                                                          collapsedToolbarSize.y + bannerHeight + 8.0f};

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
