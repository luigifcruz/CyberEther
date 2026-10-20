#ifndef JETSTREAM_RENDER_SAKURA_COMBO_HH
#define JETSTREAM_RENDER_SAKURA_COMBO_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Combo : public Component {
    struct Config {
        std::string id;
        std::vector<std::string> options;
        std::string value;
        F32 width = Fill;
        bool disabled = false;
        std::optional<U64> selectedIndex;
        Callback<void(const std::string&)> onChange;
        Callback<void(U64)> onSelect;

        bool operator==(const Config&) const = default;
    };

    Combo();
    ~Combo();

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

#endif  // JETSTREAM_RENDER_SAKURA_COMBO_HH
