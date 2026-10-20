#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_PERSIST_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_PERSIST_HH

#include "jetstream/logger.hh"
#include "jetstream/settings.hh"

#include <utility>

namespace Jetstream {

template<class Mutate>
Result PersistSettings(Mutate&& mutate) {
    Settings settings;
    JST_CHECK(Settings::Get(settings));
    std::forward<Mutate>(mutate)(settings);
    return Settings::SetAsync(settings);
}

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_ACTIONS_PERSIST_HH
