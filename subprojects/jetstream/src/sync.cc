#include "jetstream/base.hpp"

namespace Jetstream {

Result Sync::start() {
    mailbox = true;
    return Result::SUCCESS;
}

Result Sync::end() {
    mailbox = false;
    return Result::SUCCESS;
};

Result Sync::compute() {
    std::scoped_lock<std::mutex> lock(m);
    mailbox = true;

    return Result::SUCCESS;
}

Result Sync::barrier() {
    std::scoped_lock<std::mutex> lock(m);

    if (mailbox) {
        for (auto& dep : deps) {
            if ((result = dep->barrier()) != Result::SUCCESS) {
                goto end;
            }
        }
        result = this->underlyingCompute();
    }

end:
    mailbox = false;
    return result;
}

} // namespace Jetstream
