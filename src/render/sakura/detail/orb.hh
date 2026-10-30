#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_ORB_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_ORB_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>

namespace Jetstream::Sakura {

struct Orb : public Component {
    struct Config {
        Rect rect;
        Extent2D<F32> center = {0.0f, 0.0f};
        F32 radius = 0.0f;
        F32 time = 0.0f;
        F32 busy = 0.0f;
        F32 activity = 0.0f;
        F32 level = 0.0f;
        ColorRGBA<F32> listeningColor = {0.0f, 0.0f, 0.0f, 0.0f};
        ColorRGBA<F32> busyColor = {0.0f, 0.0f, 0.0f, 0.0f};
        ColorRGBA<F32> listeningAccent = {0.0f, 0.0f, 0.0f, 0.0f};
        ColorRGBA<F32> busyAccent = {0.0f, 0.0f, 0.0f, 0.0f};
        std::optional<Rect> clip;

        bool operator==(const Config&) const = default;
    };

    Orb();
    ~Orb();

    void update(Config config);
    const Config& config() const;

 protected:
    Result build(Context& ctx) override;
    Result paint() override;
    void conceal() override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_ORB_HH
