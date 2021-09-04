#ifndef JETSTREAM_ASYNC_H
#define JETSTREAM_ASYNC_H

#include "jetstream/module.hpp"

namespace Jetstream {

template<typename T>
class Async : public T {
public:
    Async(const typename T::Config & config, const typename T::Input & input) :
        T(config, input)
    {
        worker = std::thread([&]{
            while (!discard) {
                std::unique_lock<std::mutex> sync(mtx);
                access.wait(sync, [&]{ return mailbox; });

                if (!discard) {
                    result = T::compute();
                }

                mailbox = false;
                sync.unlock();
                access.notify_all();
            }
        });
    }

    ~Async() {
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

    Result compute() {
        std::unique_lock<std::mutex> sync(mtx);
        access.wait(sync, [&]{ return !mailbox; });

        if (!discard) {
            mailbox = true;
            sync.unlock();
            access.notify_all();
        }

        return Result::SUCCESS;
    }

    Result present() {
        return T::present();
    }

    Result barrier() {
        std::unique_lock<std::mutex> sync(mtx);
        access.wait(sync, [&]{ return !mailbox; });
        return result;
    }

private:
    std::thread worker;
    std::mutex mtx;
    std::condition_variable access;
    bool discard{false};
    bool mailbox{false};
    Result result;
};

} // namespace Jetstream

#endif
