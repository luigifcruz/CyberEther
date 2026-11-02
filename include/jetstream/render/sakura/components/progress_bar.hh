#ifndef JETSTREAM_RENDER_SAKURA_PROGRESS_BAR_HH
#define JETSTREAM_RENDER_SAKURA_PROGRESS_BAR_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct ProgressBar : public Component {
    struct Config {
        F32 value = 0.0f;
        std::string overlay;
        Extent2D<F32> size = {Fill, 20.0f};
        std::string colorKey = "action_btn";
        std::string backgroundColorKey;

        bool operator==(const Config&) const = default;
    };

    ProgressBar();
    ~ProgressBar();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_PROGRESS_BAR_HH
