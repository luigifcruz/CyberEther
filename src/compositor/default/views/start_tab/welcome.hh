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
    static constexpr F32 FilterPadding = Sakura::Spacing::Small;
    static constexpr F32 CategoryPadding = Sakura::Spacing::Small;
    static constexpr F32 SearchInset = Sakura::Spacing::Large;
    static constexpr F32 SearchFont = Sakura::Typography::FontSize * Sakura::Typography::Scale::Title;
    static constexpr U64 GridColumns = 3;
    static constexpr F32 MinCardWidth = 260.0f;
    static constexpr F32 ViewAllHeight = 32.0f;
    static constexpr F32 SectionIconScale = Sakura::Typography::Scale::Heading;
    static constexpr F32 SectionTitleScale = Sakura::Typography::Scale::Headline;
    static constexpr F32 SectionCaptionScale = Sakura::Typography::Scale::Large;
    static constexpr F32 SectionHeaderGap = Sakura::Spacing::XSmall;
    static constexpr F32 FooterArtTop = 96.0f;
    static constexpr F32 FooterArtBottom = 72.0f;
    static constexpr F32 FooterArtScale = 1.5f;
    static constexpr F32 FooterArtOpacity = 0.5f;
    static constexpr std::array<const char*, 6> FooterArt = {
        R"(   ______      __              ________  __             )",
        R"(  / ____/_  __/ /_  ___  _____/ ____/ /_/ /_  ___  _____)",
        R"( / /   / / / / __ \/ _ \/ ___/ __/ / __/ __ \/ _ \/ ___/)",
        R"(/ /___/ /_/ / /_/ /  __/ /  / /___/ /_/ / / /  __/ /    )",
        R"(\____/\__, /_.___/\___/_/  /_____/\__/_/ /_/\___/_/     )",
        R"(     /____/                                             )",
    };

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
        std::string accentColorKey = "cyber_blue";
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
    };

    void update(Config config) {
        this->config = std::move(config);
        updateRecent();
        updateExplore();
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
            .content = &pageFrame,
            .alwaysScrollbar = false,
            .border = false,
            .overlay = true,
            .scrollbarInset = {0.0f, Sakura::Dock::TileRounding},
        });
        tiles.update({
            .sidebar = &leftColumn,
            .page = &pageArea,
            .pageCard = this->config.pageCard,
        });
    }

    Sakura::Component& component() {
        return tiles;
    }

 private:
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

    void updateFooterArt() {
        using Text = Sakura::Text;

        footerArtTop.update({});
        footerArtBottom.update({});
        std::vector<Sakura::Stack::Item> items = {{.child = &footerArtTop, .size = FooterArtTop}};
        for (U64 index = 0; index < FooterArt.size(); ++index) {
            std::string line = FooterArt[index];
            if (index + 1 == FooterArt.size()) {
                const auto& version = config.version;
                const U64 length = std::min(version.size(), line.size());
                line.replace(line.size() - length, length, version, 0, length);
            }
            footerArtLines[index].update({
                .str = std::move(line),
                .tone = Text::Tone::Disabled,
                .align = Text::Align::Center,
                .shrinkToFit = true,
                .scale = FooterArtScale,
                .opacity = FooterArtOpacity,
            });
            items.push_back({.child = &footerArtLines[index]});
        }
        items.push_back({.child = &footerArtBottom, .size = FooterArtBottom});
        footerArt.update({
            .spacing = 0.0f,
            .items = std::move(items),
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
            .accentColorKey = example.accentColorKey,
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
                {.child = &waveSpacing, .size = WaveGap - ExploreSpacing},
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

        std::vector<Sakura::PillTabs::Item> pills = {{.label = "All"}};
        U64 selected = 0;
        for (const auto& name : config.categoryNames) {
            if (name == config.category) {
                selected = pills.size();
            }
            pills.push_back({.label = name});
        }
        categoryTabs.update({
            .items = std::move(pills),
            .selected = selected,
            .compact = true,
            .onSelect = [this](U64 index) {
                if (config.onSelectCategory) {
                    config.onSelectCategory(index == 0 || index > config.categoryNames.size()
                                                ? std::string()
                                                : config.categoryNames[index - 1]);
                }
            },
        });
        categoryFrame.update({
            .content = &categoryTabs,
            .padding = CategoryPadding,
            .border = false,
            .colorKey = "transparent",
        });
        filterDivider.update({.spacing = 0.0f});

        filterContent.update({
            .spacing = 0.0f,
            .centerCross = false,
            .items = {
                {.child = &searchFrame},
                {.child = &filterDivider},
                {.child = &categoryFrame},
            },
        });
        filterCard.update({
            .content = &filterContent,
            .rounding = Sakura::Rounding::Card,
            .border = true,
            .chrome = true,
            .colorKey = "card_surface",
            .hoveredColorKey = "card_surface",
        });
    }

    void updateExplore() {
        if (!config.progress.empty()) {
            loadingSpinner.update({});
            loadingCard.update({
                .rows = {{
                    .label = config.progress,
                    .control = &loadingSpinner,
                }},
            });
        } else if (!config.progressResult.empty()) {
            loadingResult.update({
                .str = config.progressFailed ? ICON_LC_CIRCLE_ALERT : ICON_LC_CIRCLE_CHECK,
                .colorKey = config.progressFailed ? "warning_yellow" : "success_green",
            });
            loadingCard.update({
                .rows = {{
                    .label = config.progressResult,
                    .control = &loadingResult,
                }},
            });
        }
        loadingReveal.update({
            .child = &loadingCard,
            .shown = !config.progress.empty(),
            .gap = ExploreSpacing,
            .linger = ProgressLinger,
        });
        auto items = updateFlowgraphs();
        if (items.empty() && config.progress.empty()) {
            noResults.update({.empty = config.section == WelcomeSection::Flowgraphs ? "No flowgraphs match your search."
                                                                                     : "Nothing matches your search."});
            items.push_back(&noResults);
        }
        updateHeader();
        updateFilterCard();
        filterGroup.update({
            .spacing = 0.0f,
            .centerCross = false,
            .items = {
                {.child = &filterCard},
                {.child = &loadingReveal},
            },
        });
        std::vector<Sakura::Stack::Item> stackItems = {{.child = &exploreHeader}, {.child = &filterGroup}};
        for (auto* item : items) {
            stackItems.push_back({.child = item});
        }
        updateFooterArt();
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
                .str = ICON_LC_WORKFLOW,
                .colorKey = "cyber_blue",
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
    Sakura::Divider filterDivider;
    Sakura::PillTabs categoryTabs;
    Sakura::Div categoryFrame;
    Sakura::Stack filterContent;
    Sakura::Div filterCard;
    Sakura::SettingCard noResults;
    Sakura::SettingCard loadingCard;
    Sakura::Reveal loadingReveal;
    Sakura::Stack filterGroup;
    Sakura::Spinner loadingSpinner;
    Sakura::Text loadingResult;
    Sakura::Spacer footerArtTop;
    std::array<Sakura::Text, FooterArt.size()> footerArtLines;
    Sakura::Spacer footerArtBottom;
    Sakura::Stack footerArt;
    Sakura::Stack explore;
    Sakura::Panel page;
    Sakura::Div pageFrame;
    Sakura::ScrollArea pageArea;
    SidebarTiles tiles;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_START_TAB_WELCOME_HH
