#ifndef JETSTREAM_ASYNCIFY_H
#define JETSTREAM_ASYNCIFY_H

#include "jetstream/module.hpp"

namespace Jetstream {

template<typename T>
class Asyncify : public Module {
public:
    Asyncify(const typename T::Config & config, const typename T::Input & input) {
        mod = std::make_shared<T>(config, input);
        worker = std::thread([&]{
            while (!discard) {
                std::unique_lock<std::mutex> sync(mtx);
                access.wait(sync, [&]{ return mailbox; });

                if (!discard) {
                    result = mod->compute();
                }

                mailbox = false;
                sync.unlock();
                access.notify_all();
            }
        });
    }

    ~Asyncify() {
        if (worker.joinable()) {
            {
                std::unique_lock<std::mutex> sync(mtx);
                discard = true;
                mailbox = true;
            }
            access.notify_all();
            worker.join();
        }
    }

    std::shared_ptr<T> sync() const {
        return mod;
    }

    Result compute() final {
        std::unique_lock<std::mutex> sync(mtx);
        access.wait(sync, [&]{ return !mailbox; });

        if (!discard) {
            mailbox = true;
            sync.unlock();
            access.notify_all();
        }

        return Result::SUCCESS;
    }

    Result present() final {
        return mod->present();
    }

    Result barrier() final {
        std::unique_lock<std::mutex> sync(mtx);
        access.wait(sync, [&]{ return !mailbox; });
        return result;
    }

private:
    std::shared_ptr<T> mod;
    std::thread worker;
    std::mutex mtx;
    std::condition_variable access;
    bool discard{false};
    bool mailbox{false};
    Result result;
};

} // namespace Jetstream

#endif
