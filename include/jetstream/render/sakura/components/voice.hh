#ifndef JETSTREAM_RENDER_SAKURA_VOICE_HH
#define JETSTREAM_RENDER_SAKURA_VOICE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

struct Voice : public Component {
    enum class Phase { Idle, Connecting, Listening, Busy };

    struct Config {
        Phase phase = Phase::Idle;
        F32 level = 0.0f;
        std::string startHint;
        std::string stopHint;
        Callback<void()> onStart;
        Callback<void()> onStop;

        bool operator==(const Config&) const = default;
    };

    Voice();
    ~Voice();

    void update(Config config);

 protected:
    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override;
    void layout(const Context& ctx) override;
    void wentOnscreen(U64 frame) override;
    void wentOffscreen(U64 frame) override;
    bool mouseEvent(const MouseEvent& event) override;
    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_VOICE_HH
