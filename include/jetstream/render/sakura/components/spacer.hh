#ifndef JETSTREAM_RENDER_SAKURA_SPACER_HH
#define JETSTREAM_RENDER_SAKURA_SPACER_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

namespace Jetstream::Sakura {

struct Spacer : public Component {
    struct Config {
        U64 lines = 1;

        bool operator==(const Config&) const = default;
    };

    Spacer();
    ~Spacer();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;

 private:
    Config config;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SPACER_HH
