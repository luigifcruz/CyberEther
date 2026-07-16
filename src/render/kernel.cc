#include "jetstream/render/base/kernel.hh"

namespace Jetstream::Render {

void Kernel::update() {
    updateRequested.store(true, std::memory_order_release);
}

bool Kernel::shouldEncode() const {
    return updatePrepared && !updateScheduled;
}

void Kernel::markScheduled() {
    updateScheduled = true;
}

void Kernel::commitFrame() {
    if (updateScheduled) {
        updatePrepared = false;
        updateScheduled = false;
    }
}

void Kernel::prepareForFrame() {
    updatePrepared = updateRequested.exchange(false, std::memory_order_acq_rel) ||
                     updatePrepared;
    updateScheduled = false;
}

}  // namespace Jetstream::Render
