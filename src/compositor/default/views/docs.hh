#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_DOCS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_DOCS_HH

#include "../support/docs.hh"
#include "components/sidebar_navigation.hh"
#include "components/sidebar_tiles.hh"

#include "jetstream/config.hh"
#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct DocsView {
    static constexpr const char* PaneId = "docs";
    static constexpr F32 ArticleWidth = 820.0f;
    static constexpr F32 ArticleFont = Sakura::Typography::FontSize * Sakura::Typography::Scale::Large;
    static constexpr F32 ArticlePadding = Sakura::Spacing::XXLarge;
    static constexpr F32 ChildIndent = Sakura::Spacing::Medium;
    static constexpr F32 SearchHeight = 34.0f;
#if defined(JST_OS_MAC)
    static constexpr const char* SearchShortcut = ICON_LC_COMMAND " K";
#else
    static constexpr const char* SearchShortcut = "Ctrl K";
#endif
    static constexpr F32 FooterPadding = Sakura::Spacing::Large;
    static constexpr F32 FooterCardPadding = Sakura::Spacing::Medium;
    static constexpr F32 RailWidth = 260.0f;
    static constexpr F32 RailInset = Sakura::Spacing::Large;
    static constexpr U64 OutlineTopLevel = 2;
    static constexpr U64 OutlineBottomLevel = 3;

    struct Config {
        const Docs* docs = nullptr;
        std::string page;
        std::string anchor;
        U64 navigation = 0;
        Sakura::Callback<void(const std::string&, const std::string&)> onNavigate;
        Sakura::Callback<void()> onSearch;
        Sakura::Callback<void(const std::string&)> onOpenUrl;
        Sakura::Callback<void()> onFeedback;
        Sakura::Callback<void()> onClose;
        Sakura::Callback<void()> onFocus;
    };

    void update(Config config) {
        this->config = std::move(config);
        if (!this->config.docs) {
            return;
        }
        const Docs& docs = *this->config.docs;
        const auto index = docs.find(this->config.page);
        current = index.has_value() ? &docs.pages()[*index] : nullptr;

        updateNavigation(docs);
        updateFooter();
        updateOutline();
        updateArticle(docs);

        tile.update({
            .sidebar = &sidebar,
            .page = &aside,
        });
        keys.update({
            .bindings = {
                {
                    .key = KeyCode::K,
                    .modifier = Sakura::Shortcuts::Modifier::Primary,
                    .onPressed = [this]() { search(); },
                },
            },
            .content = &tile,
        });
    }

    Sakura::Component& component() {
        return keys;
    }

    Sakura::Dock::Pane pane() {
        return {
            .id = PaneId,
            .title = "Documentation",
            .group = "main",
            .icon = ICON_LC_BOOK_OPEN,
            .content = &keys,
            .onClose = config.onClose,
            .onFocus = config.onFocus,
        };
    }

 private:
    void updateNavigation(const Docs& docs) {
        while (pageItems.size() < docs.pages().size()) {
            pageItems.push_back(std::make_unique<Sakura::SidebarItem>());
        }
        std::vector<SidebarNavigation::Section> sections;
        for (const auto& group : docs.groups()) {
            std::string heading = group.title;
            std::transform(heading.begin(), heading.end(), heading.begin(), [](unsigned char c) {
                return static_cast<char>(std::toupper(c));
            });
            SidebarNavigation::Section section{.heading = std::move(heading)};
            for (const U64 index : group.pages) {
                const auto& entry = docs.pages()[index];
                auto& item = *pageItems[index];
                item.update({
                    .label = entry.title,
                    .selected = current && current->slug == entry.slug,
                    .indent = ChildIndent * static_cast<F32>(entry.depth + 1),
                    .onSelect = [this, slug = entry.slug]() {
                        navigate(slug, "");
                    },
                });
                section.items.push_back(&item);
            }
            sections.push_back(std::move(section));
        }
        searchField.update({
            .icon = ICON_LC_SEARCH,
            .placeholder = "Search docs",
            .trailing = SearchShortcut,
            .size = {0.0f, SearchHeight},
            .onOpen = [this]() {
                search();
            },
        });
        navigation.update({
            .title = "Documentation",
            .header = &searchField,
            .sections = std::move(sections),
        });
    }

    void updateFooter() {
        using Text = Sakura::Text;

        feedbackIcon.update({
            .str = ICON_LC_MESSAGE_SQUARE_TEXT,
            .colorKey = "text_primary",
            .scale = Sakura::Typography::Scale::Lead,
        });
        feedbackLabel.update({
            .str = "Send Feedback",
            .font = Text::Font::Display,
            .colorKey = "text_primary",
            .scale = Sakura::Typography::Scale::Lead,
        });
        feedbackRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Medium,
            .items = {
                {.child = &feedbackIcon},
                {.child = &feedbackLabel},
            },
        });
        feedbackCard.update({
            .content = &feedbackRow,
            .size = {Sakura::Fill, 0.0f},
            .padding = FooterCardPadding,
            .rounding = Sakura::Rounding::Card,
            .border = true,
            .chrome = true,
            .colorKey = "card_surface",
            .hoveredColorKey = "card_surface_hovered",
            .onClick = [this]() {
                if (config.onFeedback) {
                    config.onFeedback();
                }
            },
        });
        footerFrame.update({
            .content = &feedbackCard,
            .padding = FooterPadding,
            .border = false,
            .colorKey = "transparent",
        });
        sidebar.update({
            .spacing = 0.0f,
            .centerCross = false,
            .items = {
                {.child = &navigation, .size = 1.0f, .grow = 1.0f},
                {.child = &footerFrame},
            },
        });
    }

    void updateArticle(const Docs& docs) {
        if (renderedPage != config.page || renderedDocs != &docs) {
            renderedPage = config.page;
            renderedDocs = &docs;
            source = current ? current->body : std::string();
            hero.reset();
            if (current) {
                hero = Sakura::TextMarkdown::Hero{
                    .eyebrow = current->category,
                    .title = current->title,
                    .subtitle = current->description,
                };
            }
        }
        const bool railShown = !outlineHeadings.empty() && outlineRail.visible();
        const F32 rightPadding = railShown ? RailWidth + RailInset : ArticlePadding;
        markdown.update({
            .id = "docs-article",
            .value = source,
            .fontSize = ArticleFont,
            .padding = Sakura::Padding{ArticlePadding, ArticlePadding, rightPadding, ArticlePadding},
            .scrollbar = true,
            .scrollbarInset = {0.0f, Sakura::Dock::TileRounding},
            .onLink = [this](const std::string& url) {
                return follow(url);
            },
            .hero = hero,
            .softBreaks = true,
            .maxWidth = ArticleWidth,
        });

        if (navigations != config.navigation) {
            navigations = config.navigation;
            markdown.scrollToLine(0);
            pendingAnchor = config.anchor;
            return;
        }
        if (!pendingAnchor.empty() && scrollToAnchor(pendingAnchor)) {
            pendingAnchor.clear();
        }
    }

    void updateOutline() {
        using Text = Sakura::Text;

        const auto& headings = markdown.headings();
        outlineHeadings.clear();
        std::vector<Sakura::Outline::Entry> entries;
        for (U64 index = 0; index < headings.size(); ++index) {
            if (headings[index].level >= OutlineTopLevel && headings[index].level <= OutlineBottomLevel) {
                entries.push_back({.label = headings[index].text, .level = headings[index].level - OutlineTopLevel});
                outlineHeadings.push_back(index);
            }
        }

        std::optional<U64> active;
        if (const auto heading = markdown.activeHeading(OutlineTopLevel, OutlineBottomLevel)) {
            const auto entry = std::find(outlineHeadings.begin(), outlineHeadings.end(), *heading);
            if (entry != outlineHeadings.end()) {
                active = static_cast<U64>(entry - outlineHeadings.begin());
            }
        }
        if (active != outlineActive) {
            outlineActive = active;
            ++outlineScrollRequest;
        }

        outline.update({
            .entries = std::move(entries),
            .active = outlineActive,
            .onSelect = [this](U64 index) {
                if (index < outlineHeadings.size()) {
                    markdown.revealHeading(outlineHeadings[index]);
                }
            },
        });
        outlineScroll.update({
            .id = "docs-outline",
            .content = &outline,
            .height = 0.0f,
            .alwaysScrollbar = false,
            .border = false,
            .overlay = true,
            .scrollTarget = outlineActive.has_value() ? outline.row(*outlineActive) : nullptr,
            .scrollRequest = outlineScrollRequest,
        });
        outlineTitle.update({
            .str = "ON THIS PAGE",
            .font = Text::Font::Display,
            .tone = Text::Tone::Secondary,
            .scale = Sakura::Typography::Scale::Title,
        });
        outlineStack.update({
            .spacing = Sakura::Spacing::Medium,
            .centerCross = false,
            .items = {
                {.child = &outlineTitle},
                {.child = &outlineScroll, .grow = 1.0f, .shrink = 1.0f},
            },
        });
        outlineRail.update({
            .content = &outlineStack,
            .size = {Sakura::Fill, Sakura::Fill},
            .padding = ArticlePadding,
            .border = false,
            .colorKey = "transparent",
        });
        aside.update({
            .content = &markdown,
            .rail = outlineHeadings.empty() ? nullptr : &outlineRail,
            .railWidth = RailWidth,
            .railInset = RailInset,
        });
    }

    bool follow(const std::string& url) {
        if (!url.empty() && url.front() == '#') {
            scrollToAnchor(url.substr(1));
            return true;
        }
        const auto link = Docs::Resolve(url);
        if (!link.has_value() || !config.docs) {
            return false;
        }
        if (!config.docs->find(link->slug).has_value()) {
            if (config.onOpenUrl) {
                config.onOpenUrl(Docs::WebUrl(*link));
            }
            return true;
        }
        if (current && link->slug == current->slug) {
            scrollToAnchor(link->anchor);
            return true;
        }
        navigate(link->slug, link->anchor);
        return true;
    }

    void search() {
        if (config.onSearch) {
            config.onSearch();
        }
    }

    void navigate(const std::string& slug, const std::string& anchor) {
        if (config.onNavigate) {
            config.onNavigate(slug, anchor);
        }
    }

    bool scrollToAnchor(const std::string& anchor) {
        if (anchor.empty()) {
            markdown.scrollToLine(0);
            return true;
        }
        for (const auto& heading : markdown.headings()) {
            if (Docs::Anchor(heading.text) == anchor) {
                markdown.scrollToLine(heading.line);
                return true;
            }
        }
        return false;
    }

    Config config;
    const Docs::Page* current = nullptr;
    const Docs* renderedDocs = nullptr;
    std::string renderedPage;
    std::string source;
    std::optional<Sakura::TextMarkdown::Hero> hero;
    std::string pendingAnchor;
    U64 navigations = 0;

    Sakura::PillField searchField;
    std::vector<std::unique_ptr<Sakura::SidebarItem>> pageItems;
    SidebarNavigation navigation;
    Sakura::Text feedbackIcon;
    Sakura::Text feedbackLabel;
    Sakura::Stack feedbackRow;
    Sakura::Div feedbackCard;
    Sakura::Div footerFrame;
    Sakura::Stack sidebar;

    Sakura::Markdown markdown;
    Sakura::Outline outline;
    Sakura::ScrollArea outlineScroll;
    Sakura::Text outlineTitle;
    Sakura::Stack outlineStack;
    Sakura::Div outlineRail;
    Sakura::Aside aside;
    std::vector<U64> outlineHeadings;
    std::optional<U64> outlineActive;
    U64 outlineScrollRequest = 0;

    Sakura::Shortcuts keys;
    SidebarTiles tile;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_DOCS_HH
