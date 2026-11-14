#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_STATUS_BADGE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_STATUS_BADGE_HH

#include "jetstream/render/sakura/base.hh"

#include <string>
#include <utility>

namespace Jetstream {

struct StatusBadge : public Sakura::Badge {
    enum class Tone {
        Neutral,
        Success,
        Warning,
        Error,
    };

    struct Config {
        std::string label;
        Tone tone = Tone::Neutral;
    };

    void update(Config config);

 private:
    void banner(std::string label, const char* key);
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_VIEWS_COMPONENTS_STATUS_BADGE_HH
