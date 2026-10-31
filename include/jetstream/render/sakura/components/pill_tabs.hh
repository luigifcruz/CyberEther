#ifndef JETSTREAM_RENDER_SAKURA_PILL_TABS_HH
#define JETSTREAM_RENDER_SAKURA_PILL_TABS_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct PillTabs : public Component {
    struct Item {
        std::string icon;
        std::string label;

        bool operator==(const Item&) const = default;
    };

    struct Config {
        std::vector<Item> items;
        U64 selected = 0;
        bool compact = false;
        Callback<void(U64)> onSelect;

        bool operator==(const Config&) const = default;
    };

    PillTabs();
    ~PillTabs();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_PILL_TABS_HH
