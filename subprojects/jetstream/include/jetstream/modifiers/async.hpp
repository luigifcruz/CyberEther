#ifndef JETSTREAM_ASYNC_H
#define JETSTREAM_ASYNC_H

#include <atomic>
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
                {
                    std::unique_lock<std::mutex> sync(mtx);
                    access.wait(sync, [&]{ return mailbox; });

                    if (!discard) {
                        result = T::compute();
                    }

                    mailbox = false;
                    lock = false;
                }
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
        {
            std::unique_lock<std::mutex> sync(mtx);
            mailbox = true;
            lock = true;
        }
        access.notify_all();
        return Result::SUCCESS;
    }

    Result present() {
        return T::present();
    }

    Result barrier() {
        while (lock);
        return result;
    }

private:
    std::thread worker;
    std::mutex mtx;
    std::atomic<bool> lock{false};
    std::condition_variable access;
    bool discard{false};
    bool mailbox{false};
    Result result;
};

} // namespace Jetstream

#endif
