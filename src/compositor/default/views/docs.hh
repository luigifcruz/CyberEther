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

    void update(Config config);
    Sakura::Component& component();
    Sakura::Dock::Pane pane();

 private:
    void updateNavigation(const Docs& docs);
    void updateFooter();
    void updateArticle(const Docs& docs);
    void updateOutline();
    bool follow(const std::string& url);
    void search();
    void navigate(const std::string& slug, const std::string& anchor);
    bool scrollToAnchor(const std::string& anchor);

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
