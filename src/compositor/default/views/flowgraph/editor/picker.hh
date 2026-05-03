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

struct FlowgraphBlockPickerRow : public Sakura::Component {
    struct Config {
        std::string id;
        bool selected = false;
        std::string title;
        std::string summary;
        std::function<void()> onSelect;
        std::function<void()> onCreate;
    };

    void update(Config config) {
        this->config = std::move(config);

        card.update({
            .id = this->config.id,
            .size = {-1.0f, 0.0f},
            .padding = 6.0f,
            .rounding = 8.0f,
            .border = false,
            .selected = this->config.selected,
            .colorKey = "cell_background",
            .hoveredColorKey = "cell_background",
            .selectedColorKey = "header_hovered",
            .onClick = this->config.onSelect,
            .onDoubleClick = this->config.onCreate,
        });
        row.update({.id = this->config.id + ":row", .spacing = 8.0f});
        content.update({.id = this->config.id + ":content", .spacing = 2.0f});
        icon.update({
            .id = this->config.id + ":icon",
            .str = ICON_FA_CUBE,
            .tone = Sakura::Text::Tone::Secondary,
        });
        title.update({
            .id = this->config.id + ":title",
            .str = this->config.title,
            .font = Sakura::Text::Font::Bold,
        });
        summary.update({
            .id = this->config.id + ":summary",
            .str = this->config.summary,
            .tone = Sakura::Text::Tone::Secondary,
            .wrapped = true,
            .scale = 0.85f,
        });
    }

    void render(const Sakura::Context& ctx) const {
        card.render(ctx, [this](const Sakura::Context& ctx) {
            row.render(ctx, {
                [this](const Sakura::Context& ctx) {
                    icon.render(ctx);
                },
                [this](const Sakura::Context& ctx) {
                    content.render(ctx, {
                        [this](const Sakura::Context& ctx) {
                            title.render(ctx);
                        },
                        [this](const Sakura::Context& ctx) {
                            summary.render(ctx);
                        },
                    });
                },
            });
        });
    }

 private:
    Config config;
    Sakura::Div card;
    Sakura::HStack row;
    Sakura::VStack content;
    Sakura::Text icon;
    Sakura::Text title;
    Sakura::Text summary;
};

struct FlowgraphBlockPicker : public Sakura::Component {
    struct BlockOption {
        std::string type;
        std::string title;
        std::string summary;
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
            .size = Extent2D<F32>{300.0f, 400.0f},
            .padding = Extent2D<F32>{8.0f, 8.0f},
            .rounding = 12.0f,
            .borderSize = 2.0f,
            .onClose = this->config.onClose,
        });
        layout.update({.id = this->config.id + ":layout", .spacing = 4.0f});
        title.update({
            .id = this->config.id + ":title",
            .str = "Block Picker",
            .align = Sakura::Text::Align::Center,
            .scale = 1.15f,
        });
        help.update({
            .id = this->config.id + ":help",
            .str = "Use up/down to navigate, Enter to create",
            .tone = Sakura::Text::Tone::Secondary,
            .align = Sakura::Text::Align::Center,
            .scale = 0.85f,
        });
        searchInput.update({
            .id = this->config.id + ":search",
            .value = this->config.search,
            .hint = "Search blocks...",
            .submit = Sakura::TextInput::Submit::OnEdit,
            .focus = true,
            .onChange = [this](const std::string& value) {
                if (this->config.onSearchChange) {
                    this->config.onSearchChange(value);
                }
                if (this->config.onSelectIndex) {
                    this->config.onSelectIndex(0);
                }
            },
        });
        list.update({
            .id = this->config.id + ":list",
            .size = {-1.0f, -1.0f},
            .rounding = 8.0f,
            .border = false,
            .colorKey = "card",
        });
        noMatches.update({
            .id = this->config.id + ":no-matches",
            .str = "No matching blocks.",
            .tone = Sakura::Text::Tone::Disabled,
            .align = Sakura::Text::Align::Center,
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
            },
        });

        const auto blocks = filteredBlocks();
        const int selectedIndex = clampedSelectedIndex(blocks.size());
        rows.resize(blocks.size());
        for (U64 i = 0; i < rows.size(); ++i) {
            const auto& item = blocks[i];
            rows[i].update({
                .id = this->config.id + ":block:" + item.option.type,
                .selected = static_cast<int>(i) == selectedIndex,
                .title = item.option.title,
                .summary = item.option.summary,
                .onSelect = [this, i]() {
                    if (this->config.onSelectIndex) {
                        this->config.onSelectIndex(static_cast<int>(i));
                    }
                },
                .onCreate = [this, option = item.option]() {
                    createBlock(option);
                },
            });
        }
    }

    void render(const Sakura::Context& ctx) {
        keyboard.render(ctx);

        popup.render(ctx, [this](const Sakura::Context& ctx) {
            layout.render(ctx, {
                [this](const Sakura::Context& ctx) {
                    title.render(ctx);
                },
                [this](const Sakura::Context& ctx) {
                    help.render(ctx);
                },
                [this](const Sakura::Context& ctx) {
                    searchInput.render(ctx);
                },
                [this](const Sakura::Context& ctx) {
                    list.render(ctx, [this](const Sakura::Context& ctx) {
                        for (const auto& row : rows) {
                            row.render(ctx);
                        }
                        if (rows.empty()) {
                            noMatches.render(ctx);
                        }
                    });
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

    Config config;
    Sakura::ContextMenu popup;
    Sakura::KeyboardInput keyboard;
    Sakura::VStack layout;
    Sakura::Text title;
    Sakura::Text help;
    Sakura::TextInput searchInput;
    Sakura::Div list;
    Sakura::Text noMatches;
    std::vector<FlowgraphBlockPickerRow> rows;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH
