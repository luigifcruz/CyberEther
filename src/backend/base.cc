#include "jetstream/backend/base.hh" 

namespace Jetstream::Backend {

Instance& Get() {
    static std::unique_ptr<Instance> instance;

    if (!instance) {
        instance = std::make_unique<Instance>();
    }

    return *instance;
}

}  // namespace Jetstream::Backend
