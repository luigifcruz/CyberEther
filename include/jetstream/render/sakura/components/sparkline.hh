#ifndef JETSTREAM_RENDER_SAKURA_SPARKLINE_HH
#define JETSTREAM_RENDER_SAKURA_SPARKLINE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>
#include <vector>

namespace Jetstream::Sakura {

struct Sparkline : public Component {
    struct Config {
        std::vector<F32> values;
        F32 limit = 0.0f;
        F32 alert = 0.0f;
        Extent2D<F32> size = {Fill, 28.0f};
        U64 capacity = 96;
        std::string colorKey = "text_secondary";
        std::string alertColorKey = "warning_yellow";
        std::string limitColorKey = "border";

        bool operator==(const Config&) const = default;
    };

    Sparkline();
    ~Sparkline();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SPARKLINE_HH
