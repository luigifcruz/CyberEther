#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_WELCOME_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_WELCOME_HH

#include "../../model/ui.hh"
#include "../components/assistant_composer.hh"
#include "../components/card_button.hh"
#include "../components/flowgraph_cell.hh"
#include "../components/pagination.hh"
#include "../components/section_group.hh"
#include "../components/sidebar_tiles.hh"
#include "remote.hh"

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace Jetstream {

struct WelcomeView {
    static constexpr F32 FooterCardPadding = Sakura::Spacing::Medium;
    static constexpr F32 ThemeButtonWidth = 50.0f;
    static constexpr F32 PageWidth = 1240.0f;
    static constexpr F32 FramePadding = Sakura::Spacing::Large;
    static constexpr F32 FilterHeight = 40.0f;
    static constexpr F32 WaveSpan = 0.64f;
    static constexpr F32 WaveGap = 64.0f;
    static constexpr F32 HeroTop = 48.0f;
    static constexpr F32 WordmarkEmboss = 0.0125f;
    static constexpr F32 SectionGap = Sakura::Spacing::XLarge;
    static constexpr F32 HeroButtonHeight = 40.0f;
    static constexpr F32 HeroButtonText = Sakura::Typography::Scale::Large;
    static constexpr F32 ExploreSpacing = Sakura::Spacing::Medium;
    static constexpr F64 ProgressLinger = 1.5;
    static constexpr F32 StatusHeight = 36.0f;
    static constexpr F32 StatusPadding = Sakura::Spacing::Small;
    static constexpr F32 StatusRounding = Sakura::Rounding::Card;
    static constexpr F32 StatusTuck = 2.0f;
    static constexpr F32 StatusShadeInset = 18.0f;
    static constexpr F32 StatusIconSize = 16.0f;
    static constexpr F64 StatusTransition = 0.3;
    static constexpr F32 FilterPadding = Sakura::Spacing::Small;
    static constexpr F32 CategoryPadding = Sakura::Spacing::Medium;
    static constexpr F32 CategoryCellWidth = 190.0f;
    static constexpr F32 CategoryIconScale = Sakura::Typography::Scale::Title;
    static constexpr F32 BackToTopThreshold = 600.0f;
    static constexpr F32 SearchInset = Sakura::Spacing::Large;
    static constexpr F32 SearchFont = Sakura::Typography::FontSize * Sakura::Typography::Scale::Title;
    static constexpr U64 GridColumns = 3;
    static constexpr F32 MinCardWidth = 260.0f;
    static constexpr F32 ViewAllHeight = 32.0f;
    static constexpr F32 SectionIconScale = Sakura::Typography::Scale::Heading;
    static constexpr F32 SectionTitleScale = Sakura::Typography::Scale::Headline;
    static constexpr F32 SectionCaptionScale = Sakura::Typography::Scale::Large;
    static constexpr F32 SectionHeaderGap = Sakura::Spacing::XSmall;
    static constexpr F32 FooterTop = 48.0f;
    static constexpr F32 FooterRowGap = Sakura::Spacing::Medium;
    static constexpr F32 FooterBottom = Sakura::Spacing::XLarge;
    static constexpr F32 FooterGlassEmboss = 0.03f;
    static constexpr F32 FooterGlassOpacity = 0.35f;
    static constexpr F32 FooterWordmarkScale = Sakura::Typography::Scale::Headline;

    struct Recent {
        std::string title;
        std::string path;
        bool favorite = false;
    };

    struct Example {
        std::string title;
        std::string summary;
        std::string key;
        std::string category;
        std::string diagram;
    };

    struct Category {
        std::string title;
        std::string description;
        std::vector<Example> examples;
        U64 total = 0;
    };

    struct Config {
        std::string version;
        bool assistantEnabled = true;
        bool pageCard = false;
        U64 assistantClearRequest = 0;
        std::optional<Sakura::ChatComposer::Selector> assistantSelector;
        Sakura::Callback<void(const std::string&)> onAssistantSubmit;
        Sakura::Callback<void()> onAssistantVoice;
        Sakura::Callback<void()> onNewFlowgraph;
        Sakura::Callback<void()> onOpenFlowgraph;
        Sakura::Callback<void()> onOpenTerminal;
        Sakura::Callback<void()> onOpenSettings;
        bool darkMode = true;
        Sakura::Callback<void()> onToggleTheme;
        bool sidebarOpen = true;
        std::vector<Recent> favorites;
        std::vector<Recent> recent;
        std::vector<Example> featured;
        std::vector<Category> categories;
        std::vector<std::string> categoryNames;
        std::string query;
        std::string category;
        Sakura::Callback<void(const std::string&)> onQueryChange;
        Sakura::Callback<void(const std::string&)> onSelectCategory;
        U64 page = 0;
        U64 pages = 0;
        Sakura::Callback<void(U64)> onSelectPage;
        WelcomeSection section = WelcomeSection::All;
        std::string progress;
        std::string progressResult;
        bool progressFailed = false;
        Sakura::Callback<void(WelcomeSection)> onSelectSection;
        Sakura::Callback<void(const std::string&)> onOpenRecent;
        Sakura::Callback<void(const std::string&)> onToggleFavorite;
        Sakura::Callback<void(const std::string&)> onForgetRecent;
        Sakura::Callback<void(const std::string&)> onRevealRecent;
        Sakura::Callback<void(const std::string&)> onCopyRecentPath;
        Sakura::Callback<void(const std::string&)> onOpenExample;
        std::optional<RemoteSectionView::Config> remote;
        U64 visit = 0;
    };

    void update(Config config);
    Sakura::Component& component();

 private:
    struct CategoryCell {
        Sakura::Div lead;
        Sakura::Text icon;
        Sakura::Text label;
        Sakura::Div tail;
        Sakura::Stack content;
        Sakura::Div card;
    };

    struct CategoryStyle {
        const char* title;
        const char* icon;
        const char* tintKey;
    };

    static CategoryStyle StyleOf(const std::string& category);

    struct Section {
        Sakura::Text icon;
        Sakura::Text heading;
        Sakura::Stack headingRow;
        Sakura::Text caption;
        Sakura::Stack titles;
        Sakura::Button viewAll;
        Sakura::Stack header;
        std::vector<std::unique_ptr<Sakura::FlowgraphCard>> cards;
        Sakura::Grid grid;
        Sakura::Spacer spacing;
        Sakura::Spacer headerGap;
        Sakura::Stack stack;
    };

    void updateFooter();
    void updatePageFooter();

    struct CellList {
        std::vector<std::unique_ptr<FlowgraphCell>> cells;
        FlowgraphCell empty;
        Sakura::Stack stack;
    };

    void updateCells(CellList& list, const std::vector<Recent>& entries, const std::string& empty, bool recent);
    void updateRowMenu();
    void updateRecent();
    void updateCard(Sakura::FlowgraphCard& card, const Example& example);
    void updateFeatured();
    void updateHeader();
    void updateFilterCard();
    void updateCategories();
    void updateBackToTop();
    void jumpToCategory(const std::string& name);
    bool scrollToCategory(const std::string& name);
    void updateStatusPill();
    void updateExplore();
    std::vector<Sakura::Component*> updateFlowgraphs();

    Config config;
    Pagination pager;
    Sakura::Spacer pagerSpacing;
    Sakura::Stack pagerRow;
    CellList favoriteCells;
    CellList recentCells;
    SectionGroup favoritesGroup;
    const char* assistantPrompt = AssistantComposer::RandomPrompt();
    Sakura::Menu rowMenu;
    std::string rowMenuPath;
    bool rowMenuFavorite = false;
    bool rowMenuOpen = false;
    bool rowMenuRecent = false;
    std::optional<std::tuple<std::string, bool, bool, bool>> rowMenuSynced;
    Sakura::Spacer remoteSpacing;
    RemoteSectionView remote;
    Sakura::Spacer favoritesSpacing;
    SectionGroup recentGroup;
    Sakura::Stack left;
    Sakura::Div leftFrame;
    Sakura::ScrollArea leftArea;
    Sakura::Text settingsIcon;
    Sakura::Text settingsLabel;
    Sakura::Stack settingsRow;
    Sakura::Div settingsCard;
    Sakura::Text themeIcon;
    Sakura::Div themeCard;
    Sakura::Stack actionRow;
    Sakura::Div footerFrame;
    Sakura::Stack leftColumn;
    Sakura::Text wordmark;
    Sakura::Spacer wordmarkSpacing;
    Sakura::ChatComposer assistant;
    CardButton heroNew;
    CardButton heroOpen;
    CardButton heroTerminal;
    Sakura::Grid heroActions;
    Sakura::Spacer actionsSpacing;
    Sakura::Spacer heroSpacing;
    Sakura::Stack heroColumn;
    Sakura::Spacer assistantSpacing;
    Sakura::WaveLine wave;
    Sakura::Spacer waveSpacing;
    Sakura::Stack exploreHeader;
    std::vector<std::unique_ptr<Sakura::FlowgraphCard>> featuredCards;
    Sakura::Text featuredIcon;
    Sakura::Text featuredTitle;
    Sakura::Stack featuredHeading;
    Sakura::Text featuredCaption;
    Sakura::Spacer featuredSpacing;
    Sakura::Grid featuredGrid;
    Sakura::Stack featuredContent;
    Sakura::Div featuredCard;
    std::vector<std::unique_ptr<Section>> sections;
    std::optional<std::string> pendingScroll;
    Sakura::Component* scrollTarget = nullptr;
    U64 scrollRequest = 0;
    bool scrolledDown = false;
    Sakura::Text backToTopIcon;
    Sakura::Text backToTopLabel;
    Sakura::Stack backToTopContent;
    Sakura::Div backToTop;
    Sakura::Text searchIcon;
    Sakura::TextInput search;
    Sakura::Div typeDivider;
    Sakura::PillTabs sectionTabs;
    Sakura::Spacer searchLead;
    Sakura::Spacer searchGap;
    Sakura::Spacer dividerGap;
    Sakura::Spacer tabsGap;
    Sakura::Stack searchRow;
    Sakura::Div searchFrame;
    Sakura::Div filterCard;
    std::vector<std::unique_ptr<CategoryCell>> categoryCells;
    Sakura::Grid categoryGrid;
    Sakura::SettingCard noResults;
    Sakura::Div statusLead;
    Sakura::Text statusLabel;
    Sakura::Div statusTail;
    Sakura::Stack statusContent;
    Sakura::Div statusPill;
    Sakura::Div statusLeft;
    Sakura::Div statusRight;
    Sakura::Stack statusBar;
    Sakura::Div statusShade;
    Sakura::Div statusShadeLeft;
    Sakura::Div statusShadeRight;
    Sakura::Div statusShadeStart;
    Sakura::Div statusShadeEnd;
    Sakura::Stack statusShadeInner;
    Sakura::Stack statusShadeRow;
    Sakura::Stack statusRow;
    Sakura::Reveal loadingReveal;
    Sakura::Div statusTop;
    Sakura::Stack statusSlot;
    Sakura::Stack filterGroup;
    Sakura::Spinner loadingSpinner;
    Sakura::Text loadingResult;
    Sakura::Spacer footerTop;
    Sakura::Divider footerRule;
    Sakura::Spacer footerRowGap;
    Sakura::Div footerGap;
    Sakura::Stack footerRow;
    Sakura::Text footerVersion;
    Sakura::Text footerWordmark;
    Sakura::Spacer footerBottom;
    Sakura::Stack footerArt;
    Sakura::Stack explore;
    Sakura::Panel page;
    Sakura::Div pageFrame;
    Sakura::ScrollArea pageArea;
    Sakura::Overlay pageOverlay;
    SidebarTiles tiles;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_WELCOME_HH
