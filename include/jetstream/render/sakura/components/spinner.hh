#ifndef JETSTREAM_RENDER_SAKURA_SPINNER_HH
#define JETSTREAM_RENDER_SAKURA_SPINNER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Spinner : public Component {
    struct Config {
        F32 diameter = 0.0f;
        F32 thickness = 2.0f;
        F32 period = 0.9f;
        std::string colorKey = "text_secondary";

        bool operator==(const Config&) const = default;
    };

    Spinner();
    ~Spinner();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SPINNER_HH
