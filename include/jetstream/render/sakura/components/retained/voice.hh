#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_VOICE_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_VOICE_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <functional>
#include <memory>
#include <string>

namespace Jetstream::Sakura::Retained {

struct Voice : public Component {
    enum class Phase { Idle, Connecting, Listening, Busy };

    struct Config {
        std::string id;
        Phase phase = Phase::Idle;
        F32 level = 0.0f;
        std::function<void()> onStart;
        std::function<void()> onDismiss;
    };

    Voice();
    ~Voice();

    Voice(const Voice&) = delete;
    Voice& operator=(const Voice&) = delete;

    bool update(Config config);

 protected:
    void layout(const Context& ctx) override;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_VOICE_HH
