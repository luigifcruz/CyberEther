#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_DOCS_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_DOCS_HH

#include "../support/docs.hh"
#include "components/sidebar_navigation.hh"
#include "components/sidebar_tiles.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <cctype>
#include <functional>
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
    static constexpr F32 FooterPadding = Sakura::Spacing::Large;
    static constexpr F32 FooterCardPadding = Sakura::Spacing::Medium;
    static constexpr const char* TileId = "docs-tile";

    struct Config {
        const Docs* docs = nullptr;
        std::string page;
        std::string anchor;
        U64 navigation = 0;
        std::function<void(const std::string&, const std::string&)> onNavigate;
        std::function<void(const std::string&)> onOpenUrl;
        std::function<void()> onFeedback;
        std::function<void()> onClose;
        std::function<void()> onFocus;
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
        updateArticle(docs);

        body.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 0.0f,
            .centerCross = false,
            .items = {
                {.child = &sidebar, .size = SidebarTiles::SidebarWidth},
                {.child = &markdown, .grow = 1.0f, .shrink = 1.0f},
            },
        });
        tile.update({
            .panes = {
                {
                    .id = TileId,
                    .group = TileId,
                    .content = &body,
                    .closable = false,
                    .maximizable = false,
                },
            },
            .tiles = Sakura::DockTiles{
                .weights = {0.0f},
                .children = {{.pane = TileId}},
            },
            .columns = true,
            .flexPane = TileId,
            .headers = false,
            .resizable = false,
        });
    }

    Sakura::Component& component() {
        return tile;
    }

    Sakura::Dock::Pane pane() {
        return {
            .id = PaneId,
            .title = "Documentation",
            .group = "main",
            .icon = ICON_LC_BOOK_OPEN,
            .content = &tile,
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
        navigation.update({
            .title = "Documentation",
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
        markdown.update({
            .id = "docs-article",
            .value = source,
            .fontSize = ArticleFont,
            .padding = Sakura::Padding{ArticlePadding, ArticlePadding, ArticlePadding, ArticlePadding},
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

    std::vector<std::unique_ptr<Sakura::SidebarItem>> pageItems;
    SidebarNavigation navigation;
    Sakura::Text feedbackIcon;
    Sakura::Text feedbackLabel;
    Sakura::Stack feedbackRow;
    Sakura::Div feedbackCard;
    Sakura::Div footerFrame;
    Sakura::Stack sidebar;

    Sakura::Markdown markdown;

    Sakura::Stack body;
    Sakura::Dock tile;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_DOCS_HH
