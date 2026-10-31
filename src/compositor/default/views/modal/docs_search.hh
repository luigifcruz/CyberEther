#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DOCS_SEARCH_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DOCS_SEARCH_HH

#include "../../support/docs.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"
#include "tools/text.hh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct DocsSearchView {
    static constexpr U64 ResultLimit = 6;
    static constexpr F32 Width = 720.0f;
    static constexpr F32 FieldHeight = 44.0f;
    static constexpr F32 CardPadding = 10.0f;
    static constexpr F32 CardHeight = 460.0f;
    static constexpr F32 RowPadding = 8.0f;
    static constexpr F32 RowGap = 2.0f;
    static constexpr F32 SnippetSize = 14.0f;

    struct Config {
        const Docs* docs = nullptr;
        std::string query;
        Sakura::Callback<void(const std::string&)> onQueryChange;
        Sakura::Callback<void(const std::string&, const std::string&)> onOpen;
        Sakura::Callback<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);
        if (searchedQuery != this->config.query || searchedDocs != this->config.docs) {
            searchedQuery = this->config.query;
            searchedDocs = this->config.docs;
            hits = searchedDocs ? searchedDocs->search(searchedQuery, ResultLimit) : std::vector<Docs::Hit>{};
            selected = 0;
        }

        updateField();
        updateResults();

        column.update({
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &fieldRow},
                {.child = &card},
            },
        });

        const auto submit = [this]() {
            open(selected);
        };
        keys.update({
            .bindings = {
                {.key = KeyCode::DownArrow, .repeat = true, .onPressed = [this]() { step(1); }},
                {.key = KeyCode::UpArrow, .repeat = true, .onPressed = [this]() { step(-1); }},
                {.key = KeyCode::Enter, .onPressed = submit},
                {.key = KeyCode::KeypadEnter, .onPressed = submit},
            },
            .content = &column,
        });
    }

    Sakura::Component& component() {
        return keys;
    }

    Sakura::ModalLayer::Config layer() {
        return {
            .content = &keys,
            .size = {Width, 0.0f},
            .closeButton = false,
            .chrome = false,
            .dismissOutside = true,
            .onClose = [this]() {
                close();
            },
        };
    }

 private:
    struct Result {
        Sakura::Text page;
        Sakura::Text chevron;
        Sakura::Text heading;
        Sakura::Stack title;
        Sakura::HighlightText snippet;
        Sakura::Stack content;
        Sakura::Div card;
    };

    void updateField() {
        field.update({
            .icon = ICON_LC_SEARCH,
            .placeholder = "Search docs",
            .query = config.query,
            .editing = true,
            .scale = Sakura::Typography::Scale::Large,
            .size = {0.0f, FieldHeight},
            .onChange = [this](const std::string& value) {
                if (config.onQueryChange) {
                    config.onQueryChange(value);
                }
            },
            .onCancel = [this]() {
                close();
            },
        });
        closeButton.update({
            .icon = ICON_LC_X,
            .iconColorKey = "text_secondary",
            .size = {FieldHeight, FieldHeight},
            .onClick = [this]() {
                close();
            },
        });
        fieldRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &field, .grow = 1.0f, .shrink = 1.0f},
                {.child = &closeButton},
            },
        });
    }

    void updateResults() {
        while (results.size() < hits.size()) {
            results.push_back(std::make_unique<Result>());
        }
        std::vector<Sakura::Stack::Item> items;
        for (U64 index = 0; index < hits.size(); ++index) {
            updateResult(index);
            items.push_back({.child = &results[index]->card});
        }
        const bool blank = Text::IsBlank(searchedQuery);
        empty.update({
            .str = blank ? std::string("Search the documentation.")
                         : jst::fmt::format("No results for \"{}\".", Text::TrimBlank(searchedQuery)),
            .font = Sakura::Text::Font::Display,
            .tone = Sakura::Text::Tone::Secondary,
            .align = Sakura::Text::Align::Center,
            .scale = Sakura::Typography::Scale::Caption,
        });
        emptyState.update({
            .spacing = 0.0f,
            .items = {
                {.child = &emptyTop, .grow = 1.0f},
                {.child = &empty},
                {.child = &emptyBottom, .grow = 1.0f},
            },
        });
        list.update({
            .spacing = RowGap,
            .items = std::move(items),
        });
        scroll.update({
            .id = "docs-search-results",
            .content = &list,
            .height = CardHeight - 2.0f * CardPadding,
            .alwaysScrollbar = false,
            .border = false,
        });
        card.update({
            .content = hits.empty() ? static_cast<Sakura::Component*>(&emptyState) : &scroll,
            .size = {Sakura::Fill, CardHeight},
            .padding = CardPadding,
            .rounding = FieldHeight * 0.5f,
            .chrome = true,
            .shadow = true,
            .colorKey = "pill_field_bg",
        });
    }

    void updateResult(U64 index) {
        const auto& hit = hits[index];
        auto& result = *results[index];

        result.page.update({
            .str = config.docs->pages()[hit.page].title,
            .font = Sakura::Text::Font::DisplaySemiBold,
            .scale = Sakura::Typography::Scale::Lead,
        });
        result.chevron.update({
            .str = ICON_LC_CHEVRON_RIGHT,
            .tone = Sakura::Text::Tone::Secondary,
        });
        result.heading.update({
            .str = hit.heading,
            .font = Sakura::Text::Font::DisplaySemiBold,
            .tone = Sakura::Text::Tone::Secondary,
            .scale = Sakura::Typography::Scale::Lead,
        });
        std::vector<Sakura::Stack::Item> title = {{.child = &result.page}};
        if (!hit.heading.empty()) {
            title.push_back({.child = &result.chevron});
            title.push_back({.child = &result.heading, .shrink = 1.0f});
        }
        result.title.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = std::move(title),
        });

        std::vector<Sakura::HighlightText::Mark> marks;
        for (const auto& [begin, end] : hit.marks) {
            marks.push_back({.begin = begin, .end = end});
        }
        result.snippet.update({
            .id = jst::fmt::format("docs-search:{}", index),
            .str = hit.snippet,
            .marks = std::move(marks),
            .fontSize = SnippetSize,
            .onClick = [this, index]() {
                open(index);
            },
        });

        std::vector<Sakura::Stack::Item> content = {{.child = &result.title}};
        if (!hit.snippet.empty()) {
            content.push_back({.child = &result.snippet});
        }
        result.content.update({
            .spacing = Sakura::Spacing::XSmall,
            .items = std::move(content),
        });
        result.card.update({
            .content = &result.content,
            .padding = RowPadding,
            .rounding = FieldHeight * 0.5f - CardPadding,
            .border = false,
            .selected = index == selected,
            .colorKey = "transparent",
            .hoveredColorKey = "chrome_button_hover",
            .selectedColorKey = "pill_active_bg",
            .onClick = [this, index]() {
                open(index);
            },
        });
    }

    void step(int delta) {
        const auto count = static_cast<int>(hits.size());
        if (count == 0) {
            return;
        }
        selected = static_cast<U64>((static_cast<int>(selected) + delta + count) % count);
        updateResults();
    }

    void open(U64 index) {
        if (index >= hits.size() || !config.docs || !config.onOpen) {
            return;
        }
        const auto& hit = hits[index];
        config.onOpen(config.docs->pages()[hit.page].slug, hit.anchor);
    }

    void close() {
        if (config.onClose) {
            config.onClose();
        }
    }

    Config config;
    const Docs* searchedDocs = nullptr;
    std::string searchedQuery;
    std::vector<Docs::Hit> hits;
    U64 selected = 0;

    Sakura::PillField field;
    Sakura::PillIconButton closeButton;
    Sakura::Stack fieldRow;
    std::vector<std::unique_ptr<Result>> results;
    Sakura::Text empty;
    Sakura::Spacer emptyTop;
    Sakura::Spacer emptyBottom;
    Sakura::Stack emptyState;
    Sakura::Stack list;
    Sakura::ScrollArea scroll;
    Sakura::Div card;
    Sakura::Stack column;
    Sakura::Shortcuts keys;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_MODAL_DOCS_SEARCH_HH
