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

            std::scoped_lock<std::mutex> guard(mutex);
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
        std::scoped_lock<std::mutex> guard(mutex);
        return this->underlyingPresent();
    }

protected:
    Execution& cfg;

    std::mutex mutex;
    std::launch policy;
    std::future<Result> future;

    virtual Result underlyingCompute() = 0;
    virtual Result underlyingPresent() = 0;
};

inline Result Compute(const std::vector<std::shared_ptr<Module>> modules) {
    for (const auto& transform : modules) {
        auto result = transform->compute();
        if (result != Result::SUCCESS) {
            return result;
        }
    }
    return Result::SUCCESS;
}

inline Result Barrier(const std::vector<std::shared_ptr<Module>> modules) {
    for (const auto& transform : modules) {
        auto result = transform->barrier();
        if (result != Result::SUCCESS) {
            return result;
        }
    }
    return Result::SUCCESS;
}

inline Result Present(const std::vector<std::shared_ptr<Module>> modules) {
    for (const auto& transform : modules) {
        auto result = transform->present();
        if (result != Result::SUCCESS) {
            return result;
        }
    }
    return Result::SUCCESS;
}

} // namespace Jetstream

#endif
