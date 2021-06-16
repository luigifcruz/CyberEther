#include "jetstream/base.hpp"

namespace Jetstream {

Result Async::start() {
    worker = std::thread([&]{
        while (!discard) {
            std::unique_lock<std::mutex> sync(m);
            access.wait(sync, [=]{ return mailbox || discard; });

            if (!discard) {
                for (auto& dep : deps) {
                    if ((result = dep->barrier()) != Result::SUCCESS) {
                        goto end;
                    }
                }
                result = this->underlyingCompute();
            }
    end:
            mailbox = false;
            sync.unlock();
            access.notify_all();
        }
    });

    return Result::SUCCESS;
}

Result Async::end() {
    if (worker.joinable()) {
        {
            std::scoped_lock<std::mutex> sync(m);
            discard = true;
        }
        access.notify_all();
        worker.join();
    }

    return Result::SUCCESS;
};

Result Async::compute() {
    std::unique_lock<std::mutex> sync(m);
    access.wait(sync, [=]{ return !mailbox || discard; });

    if (!discard) {
        mailbox = true;
        access.notify_all();
    }

    return Result::SUCCESS;
}

Result Async::barrier() {
    std::unique_lock<std::mutex> sync(m);
    access.wait(sync, [=]{ return !mailbox || discard; });
    return result;
}

} // namespace Jetstream
