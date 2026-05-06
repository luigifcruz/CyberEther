#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH

#include "jetstream/render/sakura/sakura.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"

#include "jetstream/types.hh"

#include <algorithm>
#include <cctype>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphBlockPicker : public Sakura::Component {
    struct BlockOption {
        struct DeviceOption {
            DeviceType device = DeviceType::CPU;
            RuntimeType runtime = RuntimeType::NATIVE;
            ProviderType provider = "generic";
        };

        std::string type;
        std::string title;
        std::string summary;
        std::string description;
        std::string category;
        std::vector<DeviceOption> devices;
        DeviceType device = DeviceType::CPU;
        RuntimeType runtime = RuntimeType::NATIVE;
        ProviderType provider = "generic";
    };

    struct Config {
        std::string id;
        std::string search;
        int selectedIndex = 0;
        std::vector<BlockOption> blocks;
        std::function<Extent2D<F32>()> onResolveGridPosition;
        std::function<void(const std::string&)> onSearchChange;
        std::function<void(int)> onSelectIndex;
        std::function<void(const std::string&, Extent2D<F32>, DeviceType, RuntimeType, ProviderType)> onCreateBlock;
        std::function<void()> onClose;
    };

    void update(Config config) {
        this->config = std::move(config);

        popup.update({
            .id = this->config.id + ":popup",
            .size = Extent2D<F32>{720.0f, 560.0f},
            .padding = Extent2D<F32>{12.0f, 12.0f},
            .rounding = 16.0f,
            .borderSize = 1.0f,
            .position = Sakura::ContextMenu::Position::ViewportCenter,
            .onClose = this->config.onClose,
        });

        mainLayout.update({
            .id = this->config.id + ":layout",
        });

        searchContainer.update({
            .id = this->config.id + ":search-box",
            .padding = 8.0f,
            .rounding = 10.0f,
            .border = true,
            .colorKey = "card",
            .borderColorKey = "border",
        });
        searchInput.update({
            .id = this->config.id + ":search",
            .value = this->config.search,
            .hint = "Search blocks...",
            .submit = Sakura::TextInput::Submit::OnEdit,
            .focus = true,
            .focusOutline = false,
            .onChange = [this](const std::string& value) {
                if (this->config.onSearchChange) {
                    this->config.onSearchChange(value);
                }
                if (this->config.onSelectIndex) {
                    this->config.onSelectIndex(0);
                }
            },
        });

        const std::vector<std::string> categories = buildCategories();
        if (std::find(categories.begin(), categories.end(), selectedCategory) == categories.end()) {
            selectedCategory = "All";
        }
        categoryButtons.resize(categories.size());
        for (size_t i = 0; i < categories.size(); ++i) {
            const bool isSelected = selectedCategory == categories[i];
            categoryButtons[i].update({
                .id = this->config.id + ":tab:" + categories[i],
                .str = categories[i],
                .variant = isSelected ? Sakura::Button::Variant::Action : Sakura::Button::Variant::Default,
                .onClick = [this, cat = categories[i]]() {
                    selectedCategory = cat;
                    if (this->config.onSelectIndex) {
                        this->config.onSelectIndex(0);
                    }
                },
            });
        }

        splitView.update({
            .id = this->config.id + ":split",
            .leftWidth = 210.0f,
            .height = 437.0f,
        });

        leftPanel.update({
            .id = this->config.id + ":left",
            .padding = 0.0f,
            .rounding = 0.0f,
            .border = false,
            .colorKey = "transparent",
            .scrollbar = true,
            .mouseScroll = true,
        });

        rightPanel.update({
            .id = this->config.id + ":right",
            .padding = 0.0f,
            .border = false,
            .colorKey = "transparent",
            .scrollbar = true,
            .mouseScroll = true,
        });

        documentationPanel.update({
            .id = this->config.id + ":documentation",
            .padding = 10.0f,
            .rounding = 10.0f,
            .border = true,
            .borderColorKey = "border",
            .colorKey = "card",
            .scrollbar = false,
            .mouseScroll = false,
        });

        rightColumnLayout.update({
            .id = this->config.id + ":right-column",
        });

        overviewPanel.update({
            .id = this->config.id + ":overview",
            .padding = 10.0f,
            .rounding = 10.0f,
            .border = true,
            .borderColorKey = "border",
            .colorKey = "card",
            .scrollbar = false,
            .mouseScroll = false,
        });
        overviewLayout.update({.id = this->config.id + ":overview-layout", .spacing = 0.0f});
        deviceRow.update({.id = this->config.id + ":devices", .spacing = 2.0f});

        const auto blocks = filteredBlocks();
        const int selectedIndex = clampedSelectedIndex(blocks.size());

        navItems.resize(blocks.size());
        for (U64 i = 0; i < navItems.size(); ++i) {
            const auto& item = blocks[i];
            const bool isSelected = static_cast<int>(i) == selectedIndex;
            navItems[i].update({
                .id = this->config.id + ":nav:" + item.option.type,
                .label = std::string(ICON_FA_CUBE) + " " + item.option.title,
                .selected = isSelected,
                .onSelect = [this, i]() {
                    if (this->config.onSelectIndex) {
                        this->config.onSelectIndex(static_cast<int>(i));
                    }
                },
            });
        }

        detailLayout.update({.id = this->config.id + ":detail"});
        if (!blocks.empty()) {
            const auto& selected = blocks[selectedIndex].option;
            overviewTitle.update({
                .id = this->config.id + ":overview-title",
                .str = std::string(ICON_FA_CUBE) + " " + selected.title,
                .scale = 1.20f,
            });
            overviewSummary.update({
                .id = this->config.id + ":overview-summary",
                .str = selected.summary,
                .tone = Sakura::Text::Tone::Secondary,
                .scale = 1.05f,
            });
            deviceButtons.resize(selected.devices.size());
            for (size_t i = 0; i < selected.devices.size(); ++i) {
                const auto device = selected.devices[i];
                deviceButtons[i].update({
                    .id = this->config.id + ":device:" + std::string(GetDeviceName(device.device)),
                    .str = std::string(GetDevicePrettyName(device.device)) + " " + ICON_FA_ARROW_RIGHT,
                    .colorKey = "action_btn",
                    .hoveredColorKey = "action_btn",
                    .activeColorKey = "action_btn",
                    .borderColorKey = "action_btn_outline",
                    .textColorKey = "action_btn_text",
                    .textScale = 1.05f,
                    .onClick = [this, option = selected, device]() {
                        BlockOption createOption = option;
                        createOption.device = device.device;
                        createOption.runtime = device.runtime;
                        createOption.provider = device.provider;
                        createBlock(createOption);
                    },
                });
            }
            detailMarkdown.update({
                .id = this->config.id + ":detail-md",
                .value = selected.description,
            });
        }

        noMatches.update({
            .id = this->config.id + ":no-matches",
            .str = "No matching blocks.",
            .tone = Sakura::Text::Tone::Disabled,
            .align = Sakura::Text::Align::Center,
        });
        noMatchesPadding.update({
            .id = this->config.id + ":no-matches-padding",
            .lines = 1,
        });

        footerText.update({
            .id = this->config.id + ":footer",
            .str = "\xE2\x86\x91\xE2\x86\x93 Navigate  \xC2\xB7  Enter Create  \xC2\xB7  Esc Close",
            .tone = Sakura::Text::Tone::Secondary,
            .align = Sakura::Text::Align::Center,
            .scale = 0.8f,
        });

        keyboard.update({
            .id = this->config.id + ":keyboard",
            .bindings = {
                {
                    .key = Sakura::KeyboardInput::Key::Down,
                    .onPressed = [this]() {
                        selectRelative(1);
                    },
                },
                {
                    .key = Sakura::KeyboardInput::Key::Up,
                    .onPressed = [this]() {
                        selectRelative(-1);
                    },
                },
                {
                    .key = Sakura::KeyboardInput::Key::Submit,
                    .onPressed = [this]() {
                        createSelectedBlock();
                    },
                },
                {
                    .key = Sakura::KeyboardInput::Key::Escape,
                    .onPressed = this->config.onClose,
                },
            },
        });
    }

    void render(const Sakura::Context& ctx) {
        keyboard.render(ctx);

        popup.render(ctx, [this](const Sakura::Context& ctx) {
            mainLayout.render(ctx, {
                [this](const Sakura::Context& ctx) {
                    searchContainer.render(ctx, [this](const Sakura::Context& ctx) {
                        searchInput.render(ctx);
                    });
                },
                [this](const Sakura::Context& ctx) {
                    std::vector<Sakura::HStack::Child> tabChildren;
                    tabChildren.reserve(categoryButtons.size());
                    for (size_t i = 0; i < categoryButtons.size(); ++i) {
                        tabChildren.push_back([this, i](const Sakura::Context& ctx) {
                            categoryButtons[i].render(ctx);
                        });
                    }
                    categoryRow.render(ctx, tabChildren);
                },
                [this](const Sakura::Context& ctx) {
                    splitView.render(ctx, {
                        [this](const Sakura::Context& ctx) {
                            leftPanel.render(ctx, [this](const Sakura::Context& ctx) {
                                if (navItems.empty()) {
                                    noMatchesPadding.render(ctx);
                                    noMatches.render(ctx);
                                } else {
                                    for (const auto& item : navItems) {
                                        item.render(ctx);
                                    }
                                }
                            });
                        },
                        [this](const Sakura::Context& ctx) {
                            if (navItems.empty()) {
                                return;
                            }
                            rightPanel.render(ctx, [this](const Sakura::Context& ctx) {
                                rightColumnLayout.render(ctx, {
                                    [this](const Sakura::Context& ctx) {
                                        overviewPanel.render(ctx, [this](const Sakura::Context& ctx) {
                                            std::vector<Sakura::VStack::Child> overviewChildren;
                                            overviewChildren.reserve(3);
                                            overviewChildren.push_back([this](const Sakura::Context& ctx) {
                                                overviewTitle.render(ctx);
                                            });
                                            overviewChildren.push_back([this](const Sakura::Context& ctx) {
                                                overviewSummary.render(ctx);
                                            });
                                            overviewChildren.push_back([this](const Sakura::Context& ctx) {
                                                std::vector<Sakura::HStack::Child> deviceChildren;
                                                deviceChildren.reserve(deviceButtons.size());
                                                for (size_t i = 0; i < deviceButtons.size(); ++i) {
                                                    deviceChildren.push_back([this, i](const Sakura::Context& ctx) {
                                                        deviceButtons[i].render(ctx);
                                                    });
                                                }
                                                deviceRow.render(ctx, deviceChildren);
                                            });
                                            overviewLayout.render(ctx, overviewChildren);
                                        });
                                    },
                                    [this](const Sakura::Context& ctx) {
                                        documentationPanel.render(ctx, [this](const Sakura::Context& ctx) {
                                            std::vector<Sakura::VStack::Child> detailChildren;
                                            detailChildren.reserve(1);
                                            detailChildren.push_back([this](const Sakura::Context& ctx) {
                                                detailMarkdown.render(ctx);
                                            });
                                            detailLayout.render(ctx, detailChildren);
                                        });
                                    },
                                });
                            });
                        },
                    });
                },
                [this](const Sakura::Context& ctx) {
                    footerText.render(ctx);
                },
            });
        });
    }

 private:
    struct BlockItem {
        BlockOption option;
        bool titleMatch = false;
    };

    static std::string normalize(const std::string& value) {
        std::string normalized = value;
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return normalized;
    }

    std::vector<std::string> buildCategories() const {
        std::vector<std::string> categories = {"All"};
        const std::vector<std::string> preferred = {"Core", "DSP", "Visualization", "IO", "Other"};

        auto hasCategory = [this](const std::string& category) {
            return std::any_of(config.blocks.begin(), config.blocks.end(), [&](const BlockOption& block) {
                return block.category == category;
            });
        };

        for (const auto& category : preferred) {
            if (hasCategory(category)) {
                categories.push_back(category);
            }
        }

        for (const auto& block : config.blocks) {
            if (block.category.empty()) {
                continue;
            }
            if (std::find(categories.begin(), categories.end(), block.category) == categories.end()) {
                categories.push_back(block.category);
            }
        }

        return categories;
    }

    std::vector<BlockItem> filteredBlocks() const {
        const std::string normalizedQuery = normalize(config.search);
        auto matchPriority = [&](const std::string& title, const std::string& summary) -> int {
            if (normalizedQuery.empty()) return 0;

            const std::string normalizedTitle = normalize(title);
            const std::string normalizedSummary = normalize(summary);
            if (normalizedTitle.find(normalizedQuery) != std::string::npos) return 0;
            if (normalizedSummary.find(normalizedQuery) != std::string::npos) return 1;
            return -1;
        };

        std::vector<BlockItem> blocks;
        for (const auto& entry : config.blocks) {
            if (selectedCategory != "All" && entry.category != selectedCategory) {
                continue;
            }
            const int priority = matchPriority(entry.title, entry.summary);
            if (priority >= 0) {
                blocks.push_back({entry, priority == 0});
            }
        }

        std::stable_sort(blocks.begin(), blocks.end(), [](const BlockItem& lhs, const BlockItem& rhs) {
            return lhs.titleMatch && !rhs.titleMatch;
        });

        return blocks;
    }

    void createBlock(const BlockOption& option) {
        if (config.onCreateBlock) {
            config.onCreateBlock(option.type,
                                 config.onResolveGridPosition(),
                                 option.device,
                                 option.runtime,
                                 option.provider);
        }
        if (config.onClose) {
            config.onClose();
        }
    }

    int clampedSelectedIndex(const U64 count) const {
        if (count == 0) {
            return 0;
        }
        if (config.selectedIndex >= static_cast<int>(count)) {
            return std::max(0, static_cast<int>(count) - 1);
        }
        return std::max(0, config.selectedIndex);
    }

    void selectRelative(const int delta) const {
        const auto blocks = filteredBlocks();
        if (blocks.empty() || !config.onSelectIndex) {
            return;
        }

        const int count = static_cast<int>(blocks.size());
        const int selectedIndex = clampedSelectedIndex(blocks.size());
        config.onSelectIndex((selectedIndex + delta + count) % count);
    }

    void createSelectedBlock() {
        const auto blocks = filteredBlocks();
        if (blocks.empty()) {
            return;
        }

        createBlock(blocks[clampedSelectedIndex(blocks.size())].option);
    }

    std::string selectedCategory = "All";
    Config config;
    Sakura::ContextMenu popup;
    Sakura::KeyboardInput keyboard;
    Sakura::VStack mainLayout;
    Sakura::Div searchContainer;
    Sakura::TextInput searchInput;
    Sakura::HStack categoryRow;
    std::vector<Sakura::Button> categoryButtons;
    Sakura::SplitView splitView;
    Sakura::Div leftPanel;
    Sakura::Div rightPanel;
    Sakura::Div documentationPanel;
    Sakura::VStack rightColumnLayout;
    Sakura::Div overviewPanel;
    Sakura::VStack overviewLayout;
    Sakura::Text overviewTitle;
    Sakura::Text overviewSummary;
    Sakura::HStack deviceRow;
    std::vector<Sakura::Button> deviceButtons;
    std::vector<Sakura::NavigationItem> navItems;
    Sakura::Spacing noMatchesPadding;
    Sakura::Text noMatches;
    Sakura::Text footerText;
    Sakura::VStack detailLayout;
    Sakura::Markdown detailMarkdown;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH
