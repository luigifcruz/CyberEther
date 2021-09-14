#ifndef JETSTREAM_ASYNC_H
#define JETSTREAM_ASYNC_H

#include <atomic>
#include "jetstream/tools/atomic.hpp"
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
                lock.wait(false, std::memory_order_acquire);

                if (!discard) {
                    result = T::compute();
                }

                lock.store(false);
                lock.notify_all();
            }
        });
    }

    ~Async() {
        if (worker.joinable()) {
            discard = true;
            lock.store(true);
            lock.notify_all();
            worker.join();
        }
    }

    Result compute() {
        lock.store(true);
        lock.notify_all();
        return Result::SUCCESS;
    }

    Result present() {
        return T::present();
    }

    Result barrier() {
        lock.wait(true, std::memory_order_acquire);
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
