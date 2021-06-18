#include "jetstream/base.hpp"

namespace Jetstream {

Result Async::start() {
    {
        std::scoped_lock<std::mutex> sync(m);
        discard = false;
        mailbox = false;
    }

    worker = std::thread([&]{
        while (!discard) {
            std::unique_lock<std::mutex> sync(m);
            access.wait(sync, [&]{ return mailbox; });

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
            mailbox = true;
        }
        access.notify_all();
        worker.join();
    }

    return Result::SUCCESS;
};

Result Async::compute() {
    std::unique_lock<std::mutex> sync(m);
    access.wait(sync, [&]{ return !mailbox; });

    if (!discard) {
        mailbox = true;
        sync.unlock();
        access.notify_all();
    }

    return Result::SUCCESS;
}

Result Async::barrier() {
    std::unique_lock<std::mutex> sync(m);
    access.wait(sync, [&]{ return !mailbox; });
    return result;
}

} // namespace Jetstream
