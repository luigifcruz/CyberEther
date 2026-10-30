#ifndef JETSTREAM_RENDER_SAKURA_TAB_BAR_HH
#define JETSTREAM_RENDER_SAKURA_TAB_BAR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct TabBar : public Component {
    struct Config {
        std::vector<std::string> labels;
        std::vector<Component*> pages;

        bool operator==(const Config&) const = default;
    };

    TabBar();
    ~TabBar();

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

#endif  // JETSTREAM_RENDER_SAKURA_TAB_BAR_HH
