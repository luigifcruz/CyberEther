#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include "type.hpp"

namespace Jetstream {

class Module {
public:
    typedef struct {
        Policy policy;
        std::vector<std::shared_ptr<Module>> producers;
    } Execution;

    explicit Module(Execution& c) : cfg(c) {
        switch (cfg.policy) {
            case Jetstream::Policy::ASYNC:
                policy = std::launch::async;
                break;
            case Jetstream::Policy::SYNC:
                policy = std::launch::deferred;
                break;
            case Jetstream::Policy::HYBRID:
                // TODO: implement load balancer
                policy = std::launch::deferred;
        }
    }
    virtual ~Module() = default;

    Result compute() {
        future = std::async(policy, [&](){
            for (auto& producer : cfg.producers) {
                auto result = producer->barrier();
                if (result != Result::SUCCESS) {
                    return result;
                }
            }

            return this->underlyingCompute();
        });

        return (future.valid()) ? Result::SUCCESS : Result::ERROR_FUTURE_INVALID;
    }

    Result barrier() {
        // TODO: this is garbage, fix error handling
        if (future.valid()) {
            future.wait();
        }
        return Result::SUCCESS;
    }

    Result present() {
        return this->underlyingPresent();
    }

protected:
    Execution& cfg;

    std::launch policy;
    std::future<Result> future;

    virtual Result underlyingCompute() = 0;
    virtual Result underlyingPresent() = 0;
};

class Engine : public std::vector<std::shared_ptr<Module>> {
public:
    Result compute() {
        if (waiting) {
            DEBUG_PUSH("compute_wait");
            std::unique_lock<std::mutex> sync(s);
            access.wait(sync);
            DEBUG_POP();
        }

        {
            const std::lock_guard<std::mutex> lock(i);
            DEBUG_PUSH("compute");

            for (const auto& transform : *this) {
                auto result = transform->compute();
                if (result != Result::SUCCESS) {
                    return result;
                }
            }

            for (const auto& transform : *this) {
                auto result = transform->barrier();
                if (result != Result::SUCCESS) {
                    return result;
                }
            }

            DEBUG_POP();
        }
        return Result::SUCCESS;
    }

    Result present() {
        waiting = true;
        {
            const std::lock_guard<std::mutex> lock(i);
            DEBUG_PUSH("present");

            for (const auto& transform : *this) {
                auto result = transform->present();
                if (result != Result::SUCCESS) {
                    return result;
                }
            }
            DEBUG_POP();
        }
        waiting = false;

        access.notify_one();
        return Result::SUCCESS;
    }

private:
    std::mutex s, i;
    std::condition_variable access;
    std::atomic<bool> waiting = false;
};

} // namespace Jetstream

#endif
