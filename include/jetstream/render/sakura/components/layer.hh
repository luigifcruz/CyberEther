#ifndef JETSTREAM_RENDER_SAKURA_LAYER_HH
#define JETSTREAM_RENDER_SAKURA_LAYER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

namespace Jetstream::Sakura {

struct Layer : public Component {
    struct Config {
        Component* content = nullptr;

        bool operator==(const Config&) const = default;
    };

    Layer();
    ~Layer();

    void update(Config config);
    virtual bool covers(const Extent2D<F32>& point) const;
    virtual bool modal() const;
    bool occludes(const Extent2D<F32>& point) const override;

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    Component* content = nullptr;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_LAYER_HH
