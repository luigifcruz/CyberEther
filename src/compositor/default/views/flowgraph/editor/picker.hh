#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include "jetstream/types.hh"
#include "tools/value.hh"

#include <algorithm>
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

    struct Config {
        Sakura::Callback<void(const std::string&, Extent2D<F32>, DeviceType, RuntimeType, ProviderType)> onCreateBlock;
    };

    void open(std::vector<BlockOption> catalog, Extent2D<F32> gridPosition) {
        blocks = std::move(catalog);
        opened = true;
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
                .icon = ICON_LC_BOX,
                .title = block.title,
                .summary = block.summary,
                .group = block.category == "Other" ? std::string() : block.category,
                .badges = std::move(badges),
                .documentation = block.description,
            });
        }
    }

    void close() {
        opened = false;
        query.clear();
        selected = 0;
    }

    bool isOpen() const {
        return opened;
    }

    void update(Config config) {
        this->config = std::move(config);

        field.update({
            .icon = ICON_LC_SEARCH,
            .placeholder = "Search blocks",
            .query = query,
            .editing = opened,
            .scale = Sakura::Typography::Scale::Large,
            .size = {0.0f, FieldHeight},
            .onChange = [this](const std::string& value) {
                query = value;
                selected = 0;
            },
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
            .field = &field,
            .docsSide = Sakura::SearchCatalog::DocsSide::Right,
            .emptyLabel = "No matching blocks.",
            .docsWidth = DocsWidth,
            .height = Height,
            .rounding = FieldHeight * 0.5f,
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
            .open = opened,
            .placement = Sakura::PopupPlacement::Inside,
            .panel = false,
            .onDismiss = [this]() {
                close();
            },
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
            .enabled = opened,
            .content = &catalog,
        });
    }

    Sakura::Component& component() {
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

    static constexpr F32 FieldHeight = 44.0f;
    static constexpr F32 DocsWidth = 585.0f;
    static constexpr F32 Height = 664.0f;

    Config config;
    bool opened = false;
    Extent2D<F32> position = {0.0f, 0.0f};
    std::vector<BlockOption> blocks;
    std::vector<Sakura::SearchCatalog::Item> items;
    std::string query;
    std::string group;
    U64 selected = 0;
    Sakura::Shortcuts keys;
    Sakura::PillField field;
    Sakura::SearchCatalog catalog;
    Sakura::Popup popup;
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH
