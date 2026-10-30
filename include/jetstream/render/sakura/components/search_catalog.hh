#ifndef JETSTREAM_RENDER_SAKURA_SEARCH_CATALOG_HH
#define JETSTREAM_RENDER_SAKURA_SEARCH_CATALOG_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct SearchCatalog : public Component {
    struct Item {
        std::string id;
        std::string icon;
        std::string title;
        std::string summary;
        std::string group;
        std::vector<std::string> badges;
        std::string documentation;

        bool operator==(const Item&) const = default;
    };

    enum class DocsSide {
        Left,
        Right,
    };

    struct Config {
        std::vector<Item> items;
        std::vector<std::string> groups;
        std::string query;
        std::string activeGroup;
        U64 selected = 0;
        Component* field = nullptr;
        DocsSide docsSide = DocsSide::Right;
        std::string emptyLabel = "No matches.";
        F32 listWidth = 300.0f;
        F32 docsWidth = 440.0f;
        F32 maxHeight = 460.0f;
        Callback<void(U64)> onSelect;
        Callback<void(const std::string&, std::optional<U64>)> onActivate;
        Callback<void(const std::string&)> onGroup;

        bool operator==(const Config&) const = default;
    };

    SearchCatalog();
    ~SearchCatalog();

    void update(Config config);
    const std::vector<std::string>& results() const;

    static std::vector<std::string> Groups(const std::vector<Item>& items, const std::vector<std::string>& groups);
    static std::vector<U64> Rank(const std::vector<Item>& items,
                                 const std::vector<std::string>& groups,
                                 const std::string& query,
                                 const std::string& activeGroup = "");

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;

    void mountParts();
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SEARCH_CATALOG_HH
