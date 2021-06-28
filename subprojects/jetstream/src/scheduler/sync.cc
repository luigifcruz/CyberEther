#include "jetstream/base.hpp"

namespace Jetstream {

Sync::Sync(const std::shared_ptr<Module> & m, const Dependencies & d) : Scheduler(m, d) {
    mailbox = true;
}

Sync::~Sync() {
    mailbox = false;
};

Result Sync::compute() {
    std::scoped_lock<std::mutex> lock(mtx);
    mailbox = true;

    return Result::SUCCESS;
}

Result Sync::barrier() {
    std::scoped_lock<std::mutex> lock(mtx);

    if (mailbox) {
        for (auto& dep : deps) {
            if ((result = dep->barrier()) != Result::SUCCESS) {
                goto end;
            }
        }
        result = mod->compute();
    }

end:
    mailbox = false;
    return result;
}

} // namespace Jetstream
