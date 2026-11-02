#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_REMOTE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_REMOTE_HH

#include "../components/card_button.hh"
#include "../components/section_group.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Jetstream {

struct RemoteSectionView {
    static constexpr F32 ButtonHeight = 36.0f;
    static constexpr F32 RowButtonSize = 28.0f;

    struct Viewer {
        std::string id;
        std::string code;
        bool waiting = false;
    };

    struct Config {
        bool started = false;
        std::vector<Viewer> viewers;
        Sakura::Callback<void()> onStart;
        Sakura::Callback<void()> onStop;
        Sakura::Callback<void()> onSettings;
        Sakura::Callback<void()> onCopyLink;
        Sakura::Callback<void()> onShowQr;
        Sakura::Callback<void(const std::string&)> onApprove;
        Sakura::Callback<void(const std::string&)> onRemove;
    };

    void update(Config config) {
        this->config = std::move(config);

        startButton.update({
            .str = ICON_LC_RADIO_TOWER "  Start Streaming",
            .height = ButtonHeight,
            .hoverTint = "green",
            .onClick = this->config.onStart,
        });
        tinted(stopButton, ICON_LC_SQUARE "  Stop Streaming", "red", this->config.onStop);
        copyButton.update({
            .str = ICON_LC_COPY "  Copy Link",
            .height = ButtonHeight,
            .onClick = this->config.onCopyLink,
        });
        qrButton.update({
            .str = ICON_LC_QR_CODE "  QR Code",
            .height = ButtonHeight,
            .onClick = this->config.onShowQr,
        });
        settingsButton.update({
            .str = ICON_LC_SETTINGS,
            .width = ButtonHeight,
            .height = ButtonHeight,
            .onClick = this->config.onSettings,
        });

        primaryRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = this->config.started ? &stopButton : &startButton, .grow = 1.0f, .shrink = 1.0f},
                {.child = &settingsButton},
            },
        });
        shareRow.update({
            .columns = 2,
            .cellPadding = {3.0f, 0.0f},
            .outerPadding = false,
            .items = {&qrButton, &copyButton},
        });
        const auto cards = updateViewers();

        std::vector<Sakura::Stack::Item> items;
        items.push_back({.child = &primaryRow});
        if (this->config.started) {
            items.push_back({.child = &shareRow});
            for (auto* card : cards) {
                items.push_back({.child = card});
            }
            if (cards.empty()) {
                items.push_back({.child = &emptyCard});
            }
        }
        body.update({
            .spacing = Sakura::Spacing::Small,
            .items = std::move(items),
        });
        group.update({
            .title = "Remote",
            .description = "Stream to another device.",
            .toolbar = &body,
            .trailing = false,
        });
    }

    Sakura::Component& component() {
        return group;
    }

 private:
    struct ViewerRow {
        std::string id;
        bool listed = false;
        Sakura::Text icon;
        Sakura::Text code;
        Sakura::Text status;
        Sakura::Stack texts;
        Sakura::Button approve;
        Sakura::Button remove;
        Sakura::Stack row;
        Sakura::Div card;
    };

    static void tinted(Sakura::Button& button, const char* label, const std::string& tone,
                       Sakura::Callback<void()> onClick) {
        button.update({
            .str = label,
            .size = {Sakura::Fill, ButtonHeight},
            .colorKey = jst::fmt::format("tint_{}_bg", tone),
            .hoveredColorKey = jst::fmt::format("tint_{}_hover", tone),
            .activeColorKey = jst::fmt::format("tint_{}_hover", tone),
            .borderColorKey = jst::fmt::format("tint_{}_border", tone),
            .textColorKey = jst::fmt::format("tint_{}_text", tone),
            .rounding = ButtonHeight * 0.5f,
            .onClick = std::move(onClick),
            .font = Sakura::Text::Font::Display,
        });
    }

    static void glyph(Sakura::Button& button, const char* icon, const char* hoverKey, Sakura::Callback<void()> onClick) {
        button.update({
            .str = icon,
            .size = {RowButtonSize, RowButtonSize},
            .colorKey = "transparent",
            .hoveredColorKey = "transparent",
            .activeColorKey = "transparent",
            .borderColorKey = "transparent",
            .textColorKey = "text_secondary",
            .hoveredTextColorKey = hoverKey,
            .padding = {0.0f, 0.0f},
            .onClick = std::move(onClick),
        });
    }

    ViewerRow& rowFor(const std::string& id, const std::unordered_set<std::string>& active) {
        for (auto& row : viewerRows) {
            if (row->id == id) {
                return *row;
            }
        }
        for (auto& row : viewerRows) {
            if (!row->listed && !active.contains(row->id)) {
                row->id = id;
                return *row;
            }
        }
        viewerRows.push_back(std::make_unique<ViewerRow>());
        viewerRows.back()->id = id;
        return *viewerRows.back();
    }

    std::vector<Sakura::Component*> updateViewers() {
        using Text = Sakura::Text;

        std::unordered_set<std::string> active;
        for (const auto& viewer : config.viewers) {
            active.insert(viewer.id);
        }
        std::vector<Sakura::Component*> cards;
        for (const auto& viewer : config.viewers) {
            auto& row = rowFor(viewer.id, active);
            cards.push_back(&row.card);
            row.icon.update({
                .str = ICON_LC_MONITOR,
                .tone = Text::Tone::Secondary,
                .scale = Sakura::Typography::Scale::Title,
            });
            row.code.update({
                .str = viewer.code,
                .font = Text::Font::Body,
                .scale = Sakura::Typography::Scale::Body,
            });
            row.status.update({
                .str = viewer.waiting ? "Waiting for approval" : "Connected",
                .font = Text::Font::Sans,
                .colorKey = viewer.waiting ? "warning_yellow" : "success_green",
                .scale = Sakura::Typography::Scale::Caption,
            });
            row.texts.update({
                .spacing = Sakura::Spacing::XSmall,
                .centerCross = false,
                .items = {{.child = &row.code}, {.child = &row.status}},
            });
            glyph(row.approve, ICON_LC_CHECK, "success_green", [this, id = viewer.id]() {
                if (config.onApprove) {
                    config.onApprove(id);
                }
            });
            glyph(row.remove, ICON_LC_X, "error_red", [this, id = viewer.id]() {
                if (config.onRemove) {
                    config.onRemove(id);
                }
            });
            std::vector<Sakura::Stack::Item> items = {
                {.child = &row.icon},
                {.child = &row.texts, .grow = 1.0f, .shrink = 1.0f},
            };
            if (viewer.waiting) {
                items.push_back({.child = &row.approve});
            }
            items.push_back({.child = &row.remove});
            row.row.update({
                .axis = Sakura::Stack::Axis::Horizontal,
                .spacing = Sakura::Spacing::Small,
                .items = std::move(items),
            });
            row.card.update({
                .content = &row.row,
                .padding = Sakura::Spacing::Medium,
                .rounding = Sakura::Rounding::Card,
                .border = true,
                .chrome = true,
                .colorKey = "card_surface",
                .hoveredColorKey = "card_surface",
            });
        }
        emptyText.update({
            .str = "No viewers yet. Share the link to invite one.",
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .align = Text::Align::Center,
            .wrapped = true,
            .scale = Sakura::Typography::Scale::Caption,
        });
        emptyCard.update({
            .content = &emptyText,
            .padding = Sakura::Spacing::Medium,
            .rounding = Sakura::Rounding::Card,
            .border = true,
            .chrome = true,
            .colorKey = "card_surface",
            .hoveredColorKey = "card_surface",
        });
        for (auto& row : viewerRows) {
            row->listed = config.started && active.contains(row->id);
        }
        return cards;
    }

    Config config;
    CardButton startButton;
    Sakura::Button stopButton;
    CardButton settingsButton;
    CardButton qrButton;
    CardButton copyButton;
    Sakura::Stack primaryRow;
    Sakura::Grid shareRow;
    std::vector<std::unique_ptr<ViewerRow>> viewerRows;
    Sakura::Text emptyText;
    Sakura::Div emptyCard;
    Sakura::Stack body;
    SectionGroup group;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_REMOTE_HH
