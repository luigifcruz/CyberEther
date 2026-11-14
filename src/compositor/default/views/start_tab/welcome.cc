#include "welcome.hh"

namespace Jetstream {

void WelcomeView::update(Config config) {
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
    leftItems.push_back({.child = &rowMenu});
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

Sakura::Component& WelcomeView::component() {
    return tiles;
}

WelcomeView::CategoryStyle WelcomeView::StyleOf(const std::string& category) {
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

void WelcomeView::updateFooter() {
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

void WelcomeView::updatePageFooter() {
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

void WelcomeView::updateCells(CellList& list, const std::vector<Recent>& entries, const std::string& empty, bool recent) {
    while (list.cells.size() < entries.size()) {
        list.cells.push_back(std::make_unique<FlowgraphCell>());
    }
    std::vector<Sakura::Stack::Item> items;
    for (U64 index = 0; index < entries.size(); ++index) {
        const auto& entry = entries[index];
        list.cells[index]->update({
            .title = entry.title,
            .favorite = entry.favorite,
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
            .onContextMenu = [this, path = entry.path, favorite = entry.favorite, recent]() {
                rowMenuPath = path;
                rowMenuFavorite = favorite;
                rowMenuRecent = recent;
                rowMenuOpen = true;
                updateRowMenu();
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

void WelcomeView::updateRowMenu() {
    const auto state = std::make_tuple(rowMenuPath, rowMenuFavorite, rowMenuRecent, rowMenuOpen);
    if (rowMenuSynced == state) {
        return;
    }
    rowMenuSynced = state;
    const auto run = [this](const Sakura::Callback<void(const std::string&)>& callback) {
        return [this, &callback]() {
            if (callback && !rowMenuPath.empty()) {
                callback(rowMenuPath);
            }
        };
    };
    std::vector<Sakura::MenuEntry> entries = {
        {.label = ICON_LC_EXTERNAL_LINK " Open", .onClick = run(config.onOpenRecent)},
        {.label = ICON_LC_FOLDER_OPEN " Show in Folder", .onClick = run(config.onRevealRecent)},
        {.label = ICON_LC_COPY " Copy Path", .onClick = run(config.onCopyRecentPath)},
        {.separator = true},
        {
            .label = rowMenuFavorite ? ICON_LC_STAR_OFF " Remove from Favorites" : ICON_LC_STAR " Add to Favorites",
            .onClick = run(config.onToggleFavorite),
        },
    };
    if (rowMenuRecent) {
        entries.push_back({.separator = true});
        entries.push_back({
            .label = ICON_LC_X " Remove from Recents",
            .colorKey = "error_red",
            .onClick = run(config.onForgetRecent),
        });
    }
    rowMenu.update({
        .entries = std::move(entries),
        .open = rowMenuOpen,
        .onClose = [this]() {
            rowMenuOpen = false;
        },
    });
}

void WelcomeView::updateRecent() {
    updateRowMenu();
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

void WelcomeView::updateCard(Sakura::FlowgraphCard& card, const Example& example) {
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

void WelcomeView::updateFeatured() {
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

void WelcomeView::updateHeader() {
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
            .acceptPlaceholder = true,
            .pickerIcon = ICON_LC_SLIDERS_HORIZONTAL,
        }, assistantPrompt),
        .selector = config.assistantSelector,
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

void WelcomeView::updateFilterCard() {
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

void WelcomeView::updateCategories() {
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

void WelcomeView::updateBackToTop() {
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

void WelcomeView::jumpToCategory(const std::string& name) {
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

bool WelcomeView::scrollToCategory(const std::string& name) {
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

void WelcomeView::updateStatusPill() {
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

void WelcomeView::updateExplore() {
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

std::vector<Sakura::Component*> WelcomeView::updateFlowgraphs() {
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

}  // namespace Jetstream
