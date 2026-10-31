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
        std::string accessed;
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
        Sakura::Callback<void(const std::string&)> onOpenExample;
        std::optional<RemoteSectionView::Config> remote;
        U64 visit = 0;
    };

    void update(Config config) {
        this->config = std::move(config);
        updateRecent();
        updateExplore();
        if (pendingScroll.has_value() && this->config.category.empty() && this->config.query.empty()) {
            scrollToCategory(*pendingScroll);
            pendingScroll.reset();
        }
        std::vector<Sakura::Stack::Item> leftItems;
        if (this->config.remote.has_value()) {
            remote.update(std::move(*this->config.remote));
            remoteSpacing.update({.lines = 1});
            leftItems.push_back({.child = &remote.component()});
            leftItems.push_back({.child = &remoteSpacing, .size = SectionGap});
        }
        leftItems.push_back({.child = &favoritesGroup});
        leftItems.push_back({.child = &favoritesSpacing, .size = SectionGap});
        leftItems.push_back({.child = &recentGroup});
        left.update({
            .spacing = 0.0f,
            .items = std::move(leftItems),
        });
        leftFrame.update({
            .content = &left,
            .padding = FramePadding,
            .border = false,
            .colorKey = "transparent",
        });
        leftArea.update({
            .id = jst::fmt::format("welcome-left:{}", this->config.visit),
            .content = &leftFrame,
            .alwaysScrollbar = false,
            .border = false,
            .scrollbarInset = {0.0f, Sakura::Dock::TileRounding},
        });
        updateFooter();
        leftColumn.update({
            .spacing = 0.0f,
            .centerCross = false,
            .items = {
                {.child = &leftArea, .size = 1.0f, .grow = 1.0f},
                {.child = &footerFrame},
            },
        });
        page.update({
            .id = "welcome-page",
            .content = &explore,
            .maxWidth = PageWidth,
            .transparent = true,
        });
        pageFrame.update({
            .content = &page,
            .padding = FramePadding,
            .border = false,
            .colorKey = "transparent",
        });
        pageArea.update({
            .id = jst::fmt::format("welcome-page:{}", this->config.visit),
            .content = &pageFrame,
            .alwaysScrollbar = false,
            .border = false,
            .overlay = true,
            .scrollbarInset = {0.0f, Sakura::Dock::TileRounding},
            .scrollTarget = scrollTarget,
            .scrollRequest = scrollRequest,
            .onScroll = [this](F32 offset) {
                scrolledDown = offset > BackToTopThreshold;
            },
        });
        updateBackToTop();
        tiles.update({
            .sidebar = &leftColumn,
            .page = &pageOverlay,
            .pageCard = this->config.pageCard,
            .sidebarOpen = this->config.sidebarOpen,
        });
    }

    Sakura::Component& component() {
        return tiles;
    }

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

    static CategoryStyle StyleOf(const std::string& category) {
        static const CategoryStyle styles[] = {
            {"Basics", ICON_LC_GRADUATION_CAP, "gallery_green"},
            {"DSP", ICON_LC_AUDIO_WAVEFORM, "gallery_purple"},
            {"Receivers", ICON_LC_RADIO, "gallery_orange"},
            {"Visualization", ICON_LC_CHART_SPLINE, "gallery_blue"},
            {"Plugins", ICON_LC_PUZZLE, "gallery_teal"},
        };
        for (const auto& style : styles) {
            if (category == style.title) {
                return style;
            }
        }
        return {"", ICON_LC_WORKFLOW, "cyber_blue"};
    }

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

    void updateFooter() {
        using Text = Sakura::Text;

        const auto card = [](Sakura::Div& div, Sakura::Component* content, Extent2D<F32> size,
                             Sakura::Callback<void()> onClick) {
            div.update({
                .content = content,
                .size = size,
                .padding = FooterCardPadding,
                .rounding = Sakura::Rounding::Card,
                .border = true,
                .chrome = true,
                .colorKey = "card_surface",
                .hoveredColorKey = onClick ? "card_surface_hovered" : "card_surface",
                .onClick = std::move(onClick),
            });
        };

        settingsIcon.update({
            .str = ICON_LC_SETTINGS,
            .colorKey = "text_primary",
            .scale = Sakura::Typography::Scale::Lead,
        });
        settingsLabel.update({
            .str = "Settings",
            .font = Text::Font::Display,
            .colorKey = "text_primary",
            .scale = Sakura::Typography::Scale::Lead,
        });
        settingsRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Medium,
            .items = {
                {.child = &settingsIcon},
                {.child = &settingsLabel},
            },
        });
        card(settingsCard, &settingsRow, {Sakura::Fill, 0.0f}, [this]() {
            if (config.onOpenSettings) {
                config.onOpenSettings();
            }
        });
        themeIcon.update({
            .str = config.darkMode ? ICON_LC_SUN : ICON_LC_MOON,
            .align = Text::Align::Center,
            .colorKey = "text_primary",
            .scale = Sakura::Typography::Scale::Lead,
        });
        card(themeCard, &themeIcon, {ThemeButtonWidth, 0.0f}, [this]() {
            if (config.onToggleTheme) {
                config.onToggleTheme();
            }
        });
        actionRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &settingsCard, .grow = 1.0f, .shrink = 1.0f},
                {.child = &themeCard},
            },
        });
        footerFrame.update({
            .content = &actionRow,
            .padding = FramePadding,
            .border = false,
            .colorKey = "transparent",
        });
    }

    void updatePageFooter() {
        using Text = Sakura::Text;

        footerTop.update({});
        footerRule.update({.spacing = 0.0f});
        footerRowGap.update({});
        footerBottom.update({});
        footerWordmark.update({
            .str = "CyberEther",
            .font = Text::Font::Display,
            .colorKey = "text_primary",
            .scale = FooterWordmarkScale,
            .emboss = FooterGlassEmboss,
            .opacity = FooterGlassOpacity,
        });
        footerVersion.update({
            .str = config.version,
            .font = Text::Font::Display,
            .tone = Text::Tone::Disabled,
            .scale = Sakura::Typography::Scale::Caption,
        });
        footerGap.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        footerRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 0.0f,
            .items = {
                {.child = &footerWordmark},
                {.child = &footerGap, .grow = 1.0f},
                {.child = &footerVersion},
            },
        });
        footerArt.update({
            .spacing = 0.0f,
            .centerCross = false,
            .items = {
                {.child = &footerTop, .size = FooterTop},
                {.child = &footerRule},
                {.child = &footerRowGap, .size = FooterRowGap},
                {.child = &footerRow},
                {.child = &footerBottom, .size = FooterBottom},
            },
        });
    }

    struct CellList {
        std::vector<std::unique_ptr<FlowgraphCell>> cells;
        FlowgraphCell empty;
        Sakura::Stack stack;
    };

    void updateCells(CellList& list, const std::vector<Recent>& entries, const std::string& empty, bool accessed) {
        while (list.cells.size() < entries.size()) {
            list.cells.push_back(std::make_unique<FlowgraphCell>());
        }
        std::vector<Sakura::Stack::Item> items;
        for (U64 index = 0; index < entries.size(); ++index) {
            const auto& entry = entries[index];
            list.cells[index]->update({
                .title = entry.title,
                .accessed = accessed ? entry.accessed : std::string(),
                .favorite = entry.favorite,
                .compact = !accessed,
                .onOpen = [this, path = entry.path]() {
                    if (config.onOpenRecent) {
                        config.onOpenRecent(path);
                    }
                },
                .onToggleFavorite = [this, path = entry.path]() {
                    if (config.onToggleFavorite) {
                        config.onToggleFavorite(path);
                    }
                },
            });
            items.push_back({.child = list.cells[index].get()});
        }
        if (entries.empty()) {
            list.empty.update({.empty = empty});
            items.push_back({.child = &list.empty});
        }
        list.stack.update({
            .spacing = Sakura::Spacing::Small,
            .items = std::move(items),
        });
    }

    void updateRecent() {
        updateCells(favoriteCells, config.favorites, "Star a recent flowgraph to keep it here.", false);
        favoritesGroup.update({
            .title = "Favorites",
            .description = "Your starred flowgraphs.",
            .toolbar = &favoriteCells.stack,
            .trailing = false,
        });
        favoritesSpacing.update({.lines = 1});
        updateCells(recentCells, config.recent, "Flowgraphs you open or save will appear here.", true);
        recentGroup.update({
            .title = "Recents",
            .description = "Recently opened flowgraphs.",
            .toolbar = &recentCells.stack,
            .trailing = false,
        });
    }

    void updateCard(Sakura::FlowgraphCard& card, const Example& example) {
        card.update({
            .id = example.key,
            .diagram = example.diagram,
            .category = example.category,
            .title = example.title,
            .description = example.summary,
            .accentColorKey = StyleOf(example.category).tintKey,
            .onOpen = [this, key = example.key]() {
                if (config.onOpenExample) {
                    config.onOpenExample(key);
                }
            },
        });
    }

    void updateFeatured() {
        using Text = Sakura::Text;

        while (featuredCards.size() < config.featured.size()) {
            featuredCards.push_back(std::make_unique<Sakura::FlowgraphCard>());
        }
        std::vector<Sakura::Component*> cards;
        for (U64 index = 0; index < config.featured.size(); ++index) {
            updateCard(*featuredCards[index], config.featured[index]);
            cards.push_back(featuredCards[index].get());
        }
        featuredIcon.update({
            .str = ICON_LC_WAND_SPARKLES,
            .colorKey = "cyber_blue",
            .scale = SectionIconScale,
        });
        featuredTitle.update({
            .str = "Featured",
            .font = Text::Font::DisplaySemiBold,
            .scale = SectionTitleScale,
        });
        featuredHeading.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &featuredIcon},
                {.child = &featuredTitle},
            },
        });
        featuredCaption.update({
            .str = "Hand-picked highlights from the gallery.",
            .font = Text::Font::Sans,
            .tone = Text::Tone::Secondary,
            .scale = SectionCaptionScale,
        });
        featuredSpacing.update({});
        featuredGrid.update({
            .columns = GridColumns,
            .minCellWidth = MinCardWidth,
            .cellPadding = {6.0f, 6.0f},
            .outerPadding = false,
            .items = std::move(cards),
        });
        featuredContent.update({
            .spacing = Sakura::Spacing::XSmall,
            .centerCross = false,
            .items = {
                {.child = &featuredHeading},
                {.child = &featuredCaption},
                {.child = &featuredSpacing, .size = SectionHeaderGap},
                {.child = &featuredGrid},
            },
        });
        featuredCard.update({
            .content = &featuredContent,
            .padding = Sakura::Spacing::XLarge,
            .rounding = Sakura::Rounding::Card,
            .border = true,
            .waves = true,
            .colorKey = "card_surface",
            .hoveredColorKey = "card_surface",
            .borderColorKey = "setting_card_border",
        });
    }

    void updateHeader() {
        using Text = Sakura::Text;

        wordmark.update({
            .str = "CyberEther",
            .font = Text::Font::Display,
            .colorKey = "cyber_blue",
            .shrinkToFit = true,
            .scale = 8.5f,
            .emboss = WordmarkEmboss,
        });
        wordmarkSpacing.update({.lines = 2});
        assistant.update({
            .id = "WelcomeAssistant",
            .style = AssistantComposer::Style({
                .icon = ICON_LC_WAND_SPARKLES,
                .backgroundColorKey = "card_surface",
                .compact = true,
                .chrome = true,
            }),
            .onSubmit = config.onAssistantSubmit,
            .onVoice = config.onAssistantVoice,
            .voiceHint = AssistantComposer::VoiceHint,
            .clearRequest = config.assistantClearRequest,
        });
        assistantSpacing.update({.lines = 3});
        wave.update({.period = 20.0f, .span = WaveSpan});
        waveSpacing.update({.lines = 3});
        heroNew.update({
            .str = ICON_LC_FILE_PLUS "  New Flowgraph",
            .height = HeroButtonHeight,
            .textScale = HeroButtonText,
            .onClick = [this]() {
                if (config.onNewFlowgraph) {
                    config.onNewFlowgraph();
                }
            },
        });
        heroOpen.update({
            .str = ICON_LC_FOLDER_OPEN "  Open Flowgraph",
            .height = HeroButtonHeight,
            .textScale = HeroButtonText,
            .onClick = [this]() {
                if (config.onOpenFlowgraph) {
                    config.onOpenFlowgraph();
                }
            },
        });
        heroTerminal.update({
            .str = ICON_LC_SQUARE_TERMINAL "  Open Terminal",
            .height = HeroButtonHeight,
            .textScale = HeroButtonText,
            .onClick = [this]() {
                if (config.onOpenTerminal) {
                    config.onOpenTerminal();
                }
            },
        });
        std::vector<Sakura::Component*> heroItems = {&heroNew, &heroOpen};
        if (config.onOpenTerminal) {
            heroItems.push_back(&heroTerminal);
        }
        heroActions.update({
            .columns = static_cast<U64>(heroItems.size()),
            .cellPadding = {4.0f, 0.0f},
            .outerPadding = false,
            .items = std::move(heroItems),
        });
        actionsSpacing.update({.lines = 3});
        std::vector<Sakura::Stack::Item> column = {{.child = &wordmark}, {.child = &wordmarkSpacing}};
        if (config.assistantEnabled) {
            column.push_back({.child = &assistant});
            column.push_back({.child = &actionsSpacing});
        }
        column.push_back({.child = &heroActions});
        heroColumn.update({
            .spacing = 0.0f,
            .widthOf = &wordmark,
            .items = std::move(column),
        });
        exploreHeader.update({
            .spacing = 0.0f,
            .items = {
                {.child = &heroSpacing, .size = HeroTop},
                {.child = &heroColumn},
                {.child = &assistantSpacing, .size = WaveGap},
                {.child = &wave},
                {.child = &waveSpacing, .size = WaveGap - ExploreSpacing - StatusHeight},
            },
        });
    }

    void updateFilterCard() {
        using Text = Sakura::Text;

        searchIcon.update({
            .str = ICON_LC_SEARCH,
            .tone = Text::Tone::Secondary,
            .scale = Sakura::Typography::Scale::Title,
        });
        search.update({
            .value = config.query,
            .hint = "Search flowgraphs...",
            .submit = Sakura::TextInput::Submit::OnEdit,
            .height = FilterHeight,
            .onChange = [this](const std::string& value) {
                if (config.onQueryChange) {
                    config.onQueryChange(value);
                }
            },
            .backgroundColorKey = "transparent",
            .padding = {4.0f, 5.0f},
            .fontSize = SearchFont,
            .fontName = Sakura::Typography::DisplaySemiboldFont,
        });
        typeDivider.update({
            .size = {1.0f, 28.0f},
            .border = false,
            .colorKey = "setting_card_border",
            .hoveredColorKey = "setting_card_border",
        });
        sectionTabs.update({
            .items = {
                {.icon = ICON_LC_LAYOUT_GRID, .label = "All"},
                {.icon = ICON_LC_WORKFLOW, .label = "Flowgraphs"},
            },
            .selected = static_cast<U64>(config.section),
            .onSelect = [this](U64 index) {
                if (config.onSelectSection) {
                    config.onSelectSection(static_cast<WelcomeSection>(index));
                }
            },
        });
        searchLead.update({});
        searchGap.update({});
        dividerGap.update({});
        tabsGap.update({});
        searchRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 0.0f,
            .items = {
                {.child = &searchLead, .size = SearchInset - FilterPadding},
                {.child = &searchIcon},
                {.child = &searchGap, .size = 12.0f},
                {.child = &search, .grow = 1.0f, .shrink = 1.0f},
                {.child = &dividerGap, .size = 12.0f},
                {.child = &typeDivider},
                {.child = &tabsGap, .size = 12.0f},
                {.child = &sectionTabs},
            },
        });
        searchFrame.update({
            .content = &searchRow,
            .padding = FilterPadding,
            .border = false,
            .colorKey = "transparent",
        });

        filterCard.update({
            .content = &searchFrame,
            .rounding = Sakura::Rounding::Card,
            .border = true,
            .chrome = true,
            .colorKey = "card_surface",
            .hoveredColorKey = "card_surface",
        });
    }

    void updateCategories() {
        using Text = Sakura::Text;

        const U64 count = config.categoryNames.size();
        while (categoryCells.size() < count) {
            categoryCells.push_back(std::make_unique<CategoryCell>());
        }
        std::vector<Sakura::Component*> cells;
        for (U64 index = 0; index < count; ++index) {
            const std::string& name = config.categoryNames[index];
            const bool selected = name == config.category;
            const auto style = StyleOf(name);
            auto& cell = *categoryCells[index];
            cell.icon.update({
                .str = style.icon,
                .align = Text::Align::Center,
                .colorKey = style.tintKey,
                .scale = CategoryIconScale,
            });
            cell.label.update({
                .str = name,
                .font = Text::Font::Display,
                .align = Text::Align::Center,
                .colorKey = "text_primary",
                .shrinkToFit = true,
                .scale = CategoryIconScale,
            });
            cell.lead.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
            cell.tail.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
            cell.content.update({
                .axis = Sakura::Stack::Axis::Horizontal,
                .spacing = Sakura::Spacing::Small,
                .items = {
                    {.child = &cell.lead, .grow = 1.0f},
                    {.child = &cell.icon},
                    {.child = &cell.label, .shrink = 1.0f},
                    {.child = &cell.tail, .grow = 1.0f},
                },
            });
            cell.card.update({
                .content = &cell.content,
                .padding = CategoryPadding,
                .rounding = Sakura::Rounding::Card,
                .border = true,
                .chrome = true,
                .selected = selected,
                .colorKey = "card_surface",
                .hoveredColorKey = "card_surface_hovered",
                .selectedColorKey = "card_surface_hovered",
                .onClick = [this, name]() {
                    jumpToCategory(name);
                },
            });
            cells.push_back(&cell.card);
        }
        categoryGrid.update({
            .columns = std::max<U64>(1, count),
            .minCellWidth = CategoryCellWidth,
            .cellPadding = {Sakura::Spacing::XSmall, Sakura::Spacing::XSmall},
            .outerPadding = false,
            .items = std::move(cells),
        });
    }

    void updateBackToTop() {
        using Text = Sakura::Text;

        backToTopIcon.update({
            .str = ICON_LC_ARROW_UP,
            .colorKey = "text_secondary",
            .scale = CategoryIconScale,
        });
        backToTopLabel.update({
            .str = "Back to top",
            .font = Text::Font::Display,
            .colorKey = "text_primary",
            .scale = CategoryIconScale,
        });
        backToTopContent.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &backToTopIcon},
                {.child = &backToTopLabel},
            },
        });
        backToTop.update({
            .content = &backToTopContent,
            .size = {0.0f, 0.0f},
            .padding = CategoryPadding,
            .rounding = Sakura::Rounding::Card,
            .border = true,
            .chrome = true,
            .shadow = true,
            .colorKey = "card_surface",
            .hoveredColorKey = "card_surface_hovered",
            .onClick = [this]() {
                scrollTarget = &pageFrame;
                ++scrollRequest;
            },
        });
        std::vector<Sakura::Overlay::Item> items = {{.child = &pageArea}};
        if (scrolledDown) {
            items.push_back({
                .child = &backToTop,
                .anchor = Sakura::Anchor::BottomRight,
                .offset = {-FramePadding, -FramePadding},
            });
        }
        pageOverlay.update({.items = std::move(items)});
    }

    void jumpToCategory(const std::string& name) {
        if (config.category.empty() && config.query.empty() && scrollToCategory(name)) {
            return;
        }
        pendingScroll = name;
        if (!config.category.empty() && config.onSelectCategory) {
            config.onSelectCategory("");
        }
        if (!config.query.empty() && config.onQueryChange) {
            config.onQueryChange("");
        }
    }

    bool scrollToCategory(const std::string& name) {
        Sakura::Component* target = nullptr;
        for (U64 index = 0; !target && index < config.categories.size() && index < sections.size(); ++index) {
            if (config.categories[index].title == name) {
                target = &sections[index]->stack;
            }
        }
        if (target) {
            scrollTarget = target;
            ++scrollRequest;
        }
        return target != nullptr;
    }

    void updateStatusPill() {
        using Text = Sakura::Text;

        const bool loading = !config.progress.empty();
        if (!loading && config.progressResult.empty()) {
            return;
        }
        const char* doneKey = config.progressFailed ? "status_warning" : "status_success";
        const char* doneBorderKey = config.progressFailed ? "status_warning_border" : "status_success_border";
        loadingSpinner.update({.diameter = StatusIconSize, .colorKey = "text_secondary"});
        loadingResult.update({
            .str = config.progressFailed ? ICON_LC_CIRCLE_ALERT : ICON_LC_CHECK,
            .colorKey = "action_btn_text",
            .scale = StatusIconSize / Sakura::Typography::FontSize,
        });
        statusLabel.update({
            .str = loading ? config.progress : config.progressResult,
            .font = Text::Font::Display,
            .colorKey = loading ? "text_primary" : "action_btn_text",
            .scale = Sakura::Typography::Scale::Lead,
        });
        statusLead.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        statusTail.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        statusContent.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = Sakura::Spacing::Small,
            .items = {
                {.child = &statusLead, .grow = 1.0f},
                {.child = loading ? static_cast<Sakura::Component*>(&loadingSpinner) : &loadingResult},
                {.child = &statusLabel},
                {.child = &statusTail, .grow = 1.0f},
            },
        });
        statusPill.update({
            .content = &statusContent,
            .size = {Sakura::Fill, StatusHeight + StatusTuck},
            .padding = StatusPadding,
            .rounding = StatusRounding,
            .corners = Sakura::Box::TopLeft | Sakura::Box::TopRight,
            .border = true,
            .colorKey = loading ? "card_surface" : doneKey,
            .hoveredColorKey = loading ? "card_surface" : doneKey,
            .borderColorKey = loading ? "tile_border" : doneBorderKey,
            .colorTransition = StatusTransition,
        });
        statusLeft.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        statusRight.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        statusBar.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 0.0f,
            .items = {
                {.child = &statusLeft, .grow = 2.0f},
                {.child = &statusPill, .grow = 1.0f},
                {.child = &statusRight, .grow = 2.0f},
            },
        });
        statusShade.update({
            .size = {Sakura::Fill, StatusTuck},
            .border = false,
            .shadow = true,
            .colorKey = "transparent",
        });
        statusShadeLeft.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        statusShadeRight.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        statusShadeStart.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        statusShadeEnd.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        statusShadeInner.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 0.0f,
            .items = {
                {.child = &statusShadeStart, .size = StatusShadeInset},
                {.child = &statusShade, .grow = 1.0f},
                {.child = &statusShadeEnd, .size = StatusShadeInset},
            },
        });
        statusShadeRow.update({
            .axis = Sakura::Stack::Axis::Horizontal,
            .spacing = 0.0f,
            .items = {
                {.child = &statusShadeLeft, .grow = 2.0f},
                {.child = &statusShadeInner, .grow = 1.0f},
                {.child = &statusShadeRight, .grow = 2.0f},
            },
        });
        statusRow.update({
            .spacing = -StatusTuck,
            .items = {
                {.child = &statusBar},
                {.child = &statusShadeRow},
            },
        });
    }

    void updateExplore() {
        updateStatusPill();
        loadingReveal.update({
            .child = &statusRow,
            .shown = !config.progress.empty(),
            .upward = true,
            .linger = ProgressLinger,
        });
        statusTop.update({.size = {0.0f, 0.0f}, .border = false, .colorKey = "transparent"});
        statusSlot.update({
            .spacing = 0.0f,
            .items = {
                {.child = &statusTop, .grow = 1.0f},
                {.child = &loadingReveal},
            },
        });
        auto items = updateFlowgraphs();
        if (items.empty() && config.progress.empty()) {
            noResults.update({.empty = config.section == WelcomeSection::Flowgraphs ? "No flowgraphs match your search."
                                                                                     : "Nothing matches your search."});
            items.push_back(&noResults);
        }
        updateHeader();
        updateFilterCard();
        updateCategories();
        filterGroup.update({
            .spacing = -StatusTuck,
            .centerCross = false,
            .items = {
                {.child = &statusSlot, .size = StatusHeight + StatusTuck},
                {.child = &filterCard},
            },
        });
        std::vector<Sakura::Stack::Item> stackItems = {
            {.child = &exploreHeader},
            {.child = &filterGroup},
            {.child = &categoryGrid},
        };
        for (auto* item : items) {
            stackItems.push_back({.child = item});
        }
        updatePageFooter();
        stackItems.push_back({.child = &footerArt});
        explore.update({
            .spacing = ExploreSpacing,
            .items = std::move(stackItems),
        });
    }

    std::vector<Sakura::Component*> updateFlowgraphs() {
        using Text = Sakura::Text;

        std::vector<Sakura::Component*> items;
        if (!config.featured.empty()) {
            updateFeatured();
            items.push_back(&featuredCard);
        }

        while (sections.size() < config.categories.size()) {
            sections.push_back(std::make_unique<Section>());
        }
        for (U64 index = 0; index < config.categories.size(); ++index) {
            const auto& category = config.categories[index];
            auto& section = *sections[index];
            while (section.cards.size() < category.examples.size()) {
                section.cards.push_back(std::make_unique<Sakura::FlowgraphCard>());
            }
            std::vector<Sakura::Component*> cards;
            for (U64 card = 0; card < category.examples.size(); ++card) {
                updateCard(*section.cards[card], category.examples[card]);
                cards.push_back(section.cards[card].get());
            }
            section.icon.update({
                .str = StyleOf(category.title).icon,
                .colorKey = StyleOf(category.title).tintKey,
                .scale = SectionIconScale,
            });
            section.heading.update({
                .str = jst::fmt::format("{} Flowgraphs", category.title),
                .font = Text::Font::DisplaySemiBold,
                .scale = SectionTitleScale,
            });
            section.headingRow.update({
                .axis = Sakura::Stack::Axis::Horizontal,
                .spacing = Sakura::Spacing::Small,
                .items = {
                    {.child = &section.icon},
                    {.child = &section.heading},
                },
            });
            section.caption.update({
                .str = category.description,
                .font = Text::Font::Sans,
                .tone = Text::Tone::Secondary,
                .scale = SectionCaptionScale,
            });
            section.grid.update({
                .columns = GridColumns,
                .minCellWidth = MinCardWidth,
                .cellPadding = {6.0f, 6.0f},
                .outerPadding = false,
                .items = std::move(cards),
            });
            section.titles.update({
                .spacing = Sakura::Spacing::XSmall,
                .centerCross = false,
                .items = {
                    {.child = &section.headingRow},
                    {.child = &section.caption},
                },
            });
            std::vector<Sakura::Stack::Item> header = {{.child = &section.titles, .grow = 1.0f, .shrink = 1.0f}};
            if (config.category.empty() && category.total > category.examples.size()) {
                section.viewAll.update({
                    .str = jst::fmt::format("View All  {}", ICON_LC_ARROW_RIGHT),
                    .size = {0.0f, ViewAllHeight},
                    .chrome = true,
                    .colorKey = "card_surface",
                    .hoveredColorKey = "card_surface_hovered",
                    .activeColorKey = "card_surface_hovered",
                    .borderColorKey = "setting_card_border",
                    .textColorKey = "text_secondary",
                    .hoveredTextColorKey = "text_primary",
                    .rounding = ViewAllHeight * 0.5f,
                    .padding = {14.0f, 0.0f},
                    .onClick = [this, title = category.title]() {
                        if (config.onSelectCategory) {
                            config.onSelectCategory(title);
                        }
                    },
                    .font = Text::Font::Sans,
                });
                header.push_back({.child = &section.viewAll});
            }
            section.header.update({
                .axis = Sakura::Stack::Axis::Horizontal,
                .spacing = Sakura::Spacing::Medium,
                .items = std::move(header),
            });
            section.spacing.update({.lines = 2});
            section.headerGap.update({});
            section.stack.update({
                .spacing = Sakura::Spacing::XSmall,
                .items = {
                    {.child = &section.spacing},
                    {.child = &section.header},
                    {.child = &section.headerGap, .size = SectionHeaderGap},
                    {.child = &section.grid},
                },
            });
            items.push_back(&section.stack);
        }
        if (!config.category.empty() && config.pages > 1) {
            pager.update({
                .page = config.page,
                .pages = config.pages,
                .onSelect = [this](U64 page) {
                    if (config.onSelectPage) {
                        config.onSelectPage(page);
                    }
                },
            });
            pagerSpacing.update({});
            pagerRow.update({
                .spacing = 0.0f,
                .centerCross = false,
                .items = {{.child = &pagerSpacing}, {.child = &pager}},
            });
            items.push_back(&pagerRow);
        }
        return items;
    }

    Config config;
    Pagination pager;
    Sakura::Spacer pagerSpacing;
    Sakura::Stack pagerRow;
    CellList favoriteCells;
    CellList recentCells;
    SectionGroup favoritesGroup;
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
