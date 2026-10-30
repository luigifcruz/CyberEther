#ifndef JETSTREAM_RENDER_SAKURA_NODE_COMBO_HH
#define JETSTREAM_RENDER_SAKURA_NODE_COMBO_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct NodeCombo : public Component {
    struct Config {
        std::string id;
        std::vector<std::string> options;
        std::string value;
        std::optional<U64> selectedIndex;
        Callback<void(const std::string&)> onChange;
        Callback<void(U64)> onSelect;

        bool operator==(const Config&) const = default;
    };

    NodeCombo();
    ~NodeCombo();

    void update(Config config);
    const Config& config() const;

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    bool mouseEvent(const MouseEvent& event) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_COMBO_HH
