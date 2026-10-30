#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_fa.hh"

#include "jetstream/types.hh"
#include "tools/value.hh"

#include <algorithm>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream {

struct FlowgraphBlockPicker {
    struct DeviceOption {
        DeviceType device = DeviceType::CPU;
        RuntimeType runtime = RuntimeType::NATIVE;
        ProviderType provider = "generic";
    };

    struct BlockOption {
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

    enum class Mode {
        Closed,
        Pill,
        Cursor,
    };

    static constexpr F32 PillWidth = 240.0f;
    static constexpr F32 PillHeight = 34.0f;
    static constexpr F32 PillMargin = 12.0f;

    struct Config {
        std::string id;
        std::function<void()> onOpenRequest;
        std::function<void(const std::string&, Extent2D<F32>, DeviceType, RuntimeType, ProviderType)> onCreateBlock;
    };

    void open(std::vector<BlockOption> catalog, Mode next, Extent2D<F32> gridPosition) {
        blocks = std::move(catalog);
        mode = next;
        position = gridPosition;
        query.clear();
        group.clear();
        selected = 0;
        items.clear();
        for (const auto& block : blocks) {
            std::vector<std::string> badges;
            for (const auto& device : block.devices) {
                badges.push_back(GetDevicePrettyName(device.device));
            }
            items.push_back({
                .id = block.type,
                .icon = ICON_FA_CUBE,
                .title = block.title,
                .summary = block.summary,
                .group = block.category == "Other" ? std::string() : block.category,
                .badges = std::move(badges),
                .documentation = block.description,
            });
        }
    }

    void close() {
        mode = Mode::Closed;
        query.clear();
        selected = 0;
    }

    bool isOpen() const {
        return mode != Mode::Closed;
    }

    void update(Config config) {
        this->config = std::move(config);
        using Pill = Sakura::PillField;

        const auto onChange = [this](const std::string& value) {
            query = value;
            selected = 0;
        };
        pill.update(Pill::Config{
            .icon = ICON_FA_MAGNIFYING_GLASS,
            .placeholder = "Search blocks",
            .query = mode == Mode::Pill ? query : std::string(),
            .editing = mode == Mode::Pill,
            .size = {PillWidth, PillHeight},
            .onOpen = this->config.onOpenRequest,
            .onChange = onChange,
            .onCancel = [this]() {
                close();
            },
        });
        cursorField.update(Pill::Config{
            .icon = ICON_FA_MAGNIFYING_GLASS,
            .placeholder = "Search blocks",
            .query = query,
            .editing = mode == Mode::Cursor,
            .size = {0.0f, PillHeight},
            .onChange = onChange,
            .onCancel = [this]() {
                close();
            },
        });

        catalog.update({
            .items = items,
            .groups = {"Core", "DSP", "Visualization", "IO"},
            .query = query,
            .activeGroup = group,
            .selected = selected,
            .field = mode == Mode::Cursor ? &cursorField : nullptr,
            .docsSide = mode == Mode::Pill ? Sakura::SearchCatalog::DocsSide::Left
                                           : Sakura::SearchCatalog::DocsSide::Right,
            .emptyLabel = "No matching blocks.",
            .onSelect = [this](U64 index) {
                selected = index;
            },
            .onActivate = [this](const std::string& id, std::optional<U64> badge) {
                createBlock(id, badge);
            },
            .onGroup = [this](const std::string& next) {
                group = next;
                selected = 0;
            },
        });
        selected = std::min<U64>(selected, catalog.results().empty() ? 0 : catalog.results().size() - 1);

        popup.update({
            .content = &keys,
            .open = isOpen(),
            .placement = mode == Mode::Pill ? Sakura::PopupPlacement::BelowEnd
                                            : Sakura::PopupPlacement::Point,
            .atCursor = mode == Mode::Cursor,
            .panel = false,
            .onDismiss = [this]() {
                close();
            },
            .passthroughAnchor = mode == Mode::Pill,
        });

        const auto submit = [this]() {
            const auto& results = catalog.results();
            if (selected < results.size()) {
                createBlock(results[selected], std::nullopt);
            }
        };
        keys.update({
            .bindings = {
                {.key = KeyCode::DownArrow, .onPressed = [this]() { step(1); }},
                {.key = KeyCode::UpArrow, .onPressed = [this]() { step(-1); }},
                {.key = KeyCode::Enter, .onPressed = submit},
                {.key = KeyCode::KeypadEnter, .onPressed = submit},
                {.key = KeyCode::Escape, .onPressed = [this]() { close(); }},
            },
            .enabled = isOpen(),
            .content = &catalog,
        });
    }

    Sakura::Component& pillComponent() {
        return pill;
    }

    Sakura::Component& popupComponent() {
        return popup;
    }

 private:
    void step(int delta) {
        const auto count = static_cast<int>(catalog.results().size());
        if (count == 0) {
            return;
        }
        selected = static_cast<U64>((static_cast<int>(selected) + delta + count) % count);
    }

    void createBlock(const std::string& type, std::optional<U64> badge) {
        const auto it = std::find_if(blocks.begin(), blocks.end(), [&](const BlockOption& block) {
            return block.type == type;
        });
        if (it == blocks.end()) {
            return;
        }
        DeviceOption target{.device = it->device, .runtime = it->runtime, .provider = it->provider};
        if (badge.has_value() && *badge < it->devices.size()) {
            target = it->devices[*badge];
        }
        const Extent2D<F32> at = position;
        close();
        if (config.onCreateBlock) {
            config.onCreateBlock(type, at, target.device, target.runtime, target.provider);
        }
    }

    Config config;
    Mode mode = Mode::Closed;
    Extent2D<F32> position = {0.0f, 0.0f};
    std::vector<BlockOption> blocks;
    std::vector<Sakura::SearchCatalog::Item> items;
    std::string query;
    std::string group;
    U64 selected = 0;
    Sakura::Shortcuts keys;
    Sakura::PillField pill;
    Sakura::PillField cursorField;
    Sakura::SearchCatalog catalog;
    Sakura::Popup popup;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH
