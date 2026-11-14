#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_FLOWGRAPH_PICKER_HH

#include "jetstream/render/sakura/base.hh"
#include "jetstream/render/tools/icons_lucide.hh"

#include "jetstream/provider.hh"
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

    void open(std::vector<BlockOption> catalog, Extent2D<F32> gridPosition);
    void close();
    bool isOpen() const;
    void update(Config config);
    Sakura::Component& component();

 private:
    void step(int delta);
    void createBlock(const std::string& type, std::optional<U64> badge);

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
