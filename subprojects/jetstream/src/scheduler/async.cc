#include "jetstream/base.hpp"

namespace Jetstream {

Async::Async(const std::shared_ptr<Module> & m, const Dependencies & d) : Scheduler(m, d) {
    {
        std::scoped_lock<std::mutex> sync(mtx);
        discard = false;
        mailbox = false;
    }

    worker = std::thread([&]{
        while (!discard) {
            std::unique_lock<std::mutex> sync(mtx);
            access.wait(sync, [&]{ return mailbox; });

            if (!discard) {
                for (auto& dep : deps) {
                    if ((result = dep->barrier()) != Result::SUCCESS) {
                        goto end;
                    }
                }
                result = mod->compute();
            }
    end:
            mailbox = false;
            sync.unlock();
            access.notify_all();
        }
    });
}

Async::~Async() {
    if (worker.joinable()) {
        {
            std::scoped_lock<std::mutex> sync(mtx);
            discard = true;
            mailbox = true;
        }
        access.notify_all();
        worker.join();
    }
};

Result Async::compute() {
    std::unique_lock<std::mutex> sync(mtx);
    access.wait(sync, [&]{ return !mailbox; });

    if (!discard) {
        mailbox = true;
        sync.unlock();
        access.notify_all();
    }

    return Result::SUCCESS;
}

Result Async::barrier() {
    std::unique_lock<std::mutex> sync(mtx);
    access.wait(sync, [&]{ return !mailbox; });
    return result;
}

} // namespace Jetstream
