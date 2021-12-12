#ifndef JETSTREAM_ASYNC_H
#define JETSTREAM_ASYNC_H

#include <atomic>
#include "jetstream/module.hpp"

namespace Jetstream {

template<typename T>
class Async : public T {
public:
    Async(const typename T::Config& config, const typename T::Input& input) :
        T(config, input)
    {
        worker = std::thread([&]{
            while (!discard.load()) {
                std::unique_lock<std::mutex> lock(mtx);
                while (!mailbox.load(std::memory_order_relaxed)) {
                    access.wait(lock);
                }

                if (!discard.load()) {
                    result = T::compute();
                }

                mailbox.store(false, std::memory_order_release);
            }
        });
    }

    ~Async() {
        if (worker.joinable()) {
            discard.store(true);
            this->compute();
            worker.join();
        }
    }

    Result compute() {
        {
            std::unique_lock<std::mutex> lock(mtx);
            mailbox.store(true, std::memory_order_release);
        }
        access.notify_all();
        return Result::SUCCESS;
    }

    Result present() {
        return T::present();
    }

    Result barrier() {
        while (mailbox.load(std::memory_order_relaxed)) {
#if defined __i386__ || defined __x86_64__
            __builtin_ia32_pause();
#endif
        }
        return result;
    }

private:
    Result result;
    std::mutex mtx;
    std::thread worker;
    std::condition_variable access;
    std::atomic<bool> mailbox{false};
    std::atomic<bool> discard{false};
};

} // namespace Jetstream

#endif
