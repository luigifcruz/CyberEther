#ifndef JETSTREAM_RENDER_SAKURA_FLOWGRAPH_CARD_HH
#define JETSTREAM_RENDER_SAKURA_FLOWGRAPH_CARD_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct FlowgraphCard : public Component {
    struct Config {
        std::string id;
        std::string diagram;
        std::string category;
        std::string title;
        std::string description;
        std::string accentColorKey = "cyber_blue";
        Callback<void()> onOpen;

        bool operator==(const Config&) const = default;
    };

    FlowgraphCard();
    ~FlowgraphCard();

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

#endif  // JETSTREAM_RENDER_SAKURA_FLOWGRAPH_CARD_HH
