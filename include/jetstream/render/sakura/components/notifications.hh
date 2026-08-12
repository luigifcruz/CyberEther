#ifndef JETSTREAM_RENDER_SAKURA_NOTIFICATIONS_HH
#define JETSTREAM_RENDER_SAKURA_NOTIFICATIONS_HH

#include <jetstream/render/sakura/component.hh>
#include <jetstream/types.hh>

#include <memory>
#include <string>

namespace Jetstream::Sakura {

struct Notifications {
    struct Config {
        std::string id;
        std::string backgroundColorKey = "notification_bg";
        F32 rounding = 12.0f;
    };

    Notifications();
    ~Notifications();

    Notifications(Notifications&&) noexcept;
    Notifications& operator=(Notifications&&) noexcept;

    Notifications(const Notifications&) = delete;
    Notifications& operator=(const Notifications&) = delete;

    bool update(Config config);
    void render(const Context& ctx) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NOTIFICATIONS_HH
