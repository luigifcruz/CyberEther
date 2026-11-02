#ifndef JETSTREAM_RENDER_SAKURA_WAVE_LINE_HH
#define JETSTREAM_RENDER_SAKURA_WAVE_LINE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>

namespace Jetstream::Sakura {

struct WaveLine : public Component {
    struct Config {
        F32 period = 18.0f;
        F32 span = 1.0f;

        bool operator==(const Config&) const = default;
    };

    WaveLine();
    ~WaveLine();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_WAVE_LINE_HH
