#ifndef JETSTREAM_RENDER_SAKURA_BADGE_HH
#define JETSTREAM_RENDER_SAKURA_BADGE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Badge : public Component {
    struct Config {
        std::string label;
        std::string backgroundColorKey = "header_hovered";
        std::string borderColorKey = "border";
        std::string textColorKey = "text_primary";

        bool operator==(const Config&) const = default;
    };

    Badge();
    ~Badge();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_BADGE_HH
