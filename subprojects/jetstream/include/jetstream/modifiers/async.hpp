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
#if __cpp_lib_atomic_wait
                lock.wait(false, std::memory_order_acquire);
#else
                std::unique_lock<std::mutex> sync(mtx);
                access.wait(sync, [&]{ return mailbox; });
#endif

                if (!discard) {
                    result = T::compute();
                }

#if __cpp_lib_atomic_wait
                lock.store(false, std::memory_order_release);
                lock.notify_all();
#else
                mailbox = false;
                sync.unlock();
                access.notify_all();
#endif
            }
        });
    }

    ~Async() {
        if (worker.joinable()) {
            discard = true;
            this->compute();
            worker.join();
        }
    }

    Result compute() {
#if __cpp_lib_atomic_wait
        lock.store(true, std::memory_order_release);
        lock.notify_all();
#else
        {
            std::unique_lock<std::mutex> sync(mtx);
            mailbox = true;
        }
        access.notify_all();
#endif
        return Result::SUCCESS;
    }

    Result present() {
        return T::present();
    }

    Result barrier() {
#if __cpp_lib_atomic_wait
        lock.wait(true, std::memory_order_acquire);
#else
        std::unique_lock<std::mutex> sync(mtx);
        access.wait(sync, [&]{ return !mailbox; });
#endif
        return result;
    }

private:
    Result result;
    std::thread worker;
    bool discard{false};

#if __cpp_lib_atomic_wait
    alignas(4) std::atomic<bool> lock{false};
#else
    std::mutex mtx;
    std::condition_variable access;
    bool mailbox = false;
#endif
};

} // namespace Jetstream

#endif
