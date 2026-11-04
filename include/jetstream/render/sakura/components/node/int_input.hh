#ifndef JETSTREAM_RENDER_SAKURA_NODE_INT_INPUT_HH
#define JETSTREAM_RENDER_SAKURA_NODE_INT_INPUT_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct NodeIntInput : public Component {
    struct Config {
        std::string id;
        I64 value = 0;
        std::string unit;
        Callback<void(I64)> onChange;

        bool operator==(const Config&) const = default;
    };

    NodeIntInput();
    ~NodeIntInput();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_INT_INPUT_HH
