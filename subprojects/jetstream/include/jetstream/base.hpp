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
            case Jetstream::Policy::SYNC:
                policy = std::launch::deferred;
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
        {
            const std::lock_guard<std::mutex> lock(i);
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
        }

        access.notify_all();
        return Result::SUCCESS;
    }

    Result present() {
        std::unique_lock<std::mutex> sync(s);
        access.wait(sync);

        {
            const std::lock_guard<std::mutex> lock(i);

            for (const auto& transform : *this) {
                auto result = transform->present();
                if (result != Result::SUCCESS) {
                    return result;
                }
            }
        }

        return Result::SUCCESS;
    }

private:
    std::mutex s, i;
    std::condition_variable access;
};

} // namespace Jetstream

#endif
