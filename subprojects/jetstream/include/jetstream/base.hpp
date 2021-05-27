#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include "type.hpp"

namespace Jetstream {

class Module {
public:
    explicit Module(std::shared_ptr<Module> producer = nullptr) : producer(producer) {};
    virtual ~Module() = default;

    Result compute(bool async = true) {
        auto mode = (async) ? std::launch::async : std::launch::deferred;
        future = std::async(mode, [&](){
            if (producer) {
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
    std::mutex mutex;
    std::future<Result> future;
    std::shared_ptr<Module> producer;

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
