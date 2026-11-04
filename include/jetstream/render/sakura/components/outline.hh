#ifndef JETSTREAM_RENDER_SAKURA_OUTLINE_HH
#define JETSTREAM_RENDER_SAKURA_OUTLINE_HH

#include <jetstream/render/sakura/callback.hh>
#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Outline : public Component {
    struct Entry {
        std::string label;
        U64 level = 0;

        bool operator==(const Entry&) const = default;
    };

    struct Config {
        std::vector<Entry> entries;
        std::optional<U64> active;
        Callback<void(U64)> onSelect;

        bool operator==(const Config&) const = default;
    };

    Outline();
    ~Outline();

    void update(Config config);
    Component* row(U64 index) const;

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

#endif  // JETSTREAM_RENDER_SAKURA_OUTLINE_HH
