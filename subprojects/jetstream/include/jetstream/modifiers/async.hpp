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
                while (!lock.load(std::memory_order_relaxed)) {
                    __builtin_ia32_pause();
                }
#endif
                if (!discard) {
                    result = T::compute();
                }

                lock.store(false, std::memory_order_release);
#if __cpp_lib_atomic_wait
                lock.notify_all();
#endif
            }
        });
    }

    ~Async() {
        if (worker.joinable()) {
            discard = true;
            lock.store(true, std::memory_order_release);
#if __cpp_lib_atomic_wait
            lock.notify_all();
#endif
            worker.join();
        }
    }

    Result compute() {
        lock.store(true, std::memory_order_release);
#if __cpp_lib_atomic_wait
        lock.notify_all();
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
        while (lock.load(std::memory_order_relaxed)) {
            __builtin_ia32_pause();
        }
#endif
        return result;
    }

private:
    Result result;
    std::thread worker;
    bool discard{false};
    alignas(4) std::atomic<bool> lock{false};
};

} // namespace Jetstream

#endif
