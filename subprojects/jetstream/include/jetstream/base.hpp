#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include <complex>
#include <future>
#include <iostream>
#include <vector>

#include "jetstream_config.hpp"

namespace Jetstream {

enum Result {
    SUCCESS = 0,
    ERROR = 1,
    ERROR_FUTURE_INVALID,
};

class Transform {
public:
    explicit Transform() {};
    virtual ~Transform() = default;

    Result compute(std::shared_ptr<Transform> input = nullptr, bool async = true) {
        auto mode = (async) ? std::launch::async : std::launch::deferred;
        future = std::async(mode, [=](){
            if (input) {
                input->barrier();
            }

            std::lock_guard<std::mutex> guard(mutex);
            return this->underlyingCompute();
        });

        return (future.valid()) ? Result::SUCCESS : Result::ERROR_FUTURE_INVALID;
    }

    Result barrier() {
        if (future.valid()) {
            return future.get();
        }
        return Result::ERROR_FUTURE_INVALID;
    }

    Result present() {
        if (this->barrier() == Result::SUCCESS) {
            std::lock_guard<std::mutex> guard(mutex);
            return this->underlyingPresent();
        }
        return Result::ERROR_FUTURE_INVALID;
    }

protected:
    std::mutex mutex;
    std::future<Result> future;

    virtual Result underlyingCompute() = 0;
    virtual Result underlyingPresent() = 0;
};

class Executor {
public:
inline static Result Barrier(const std::vector<std::shared_ptr<Transform>> transforms) {
    for (const auto& transform : transforms) {
        if (transform->barrier() != Result::SUCCESS) {
            return Result::ERROR;
        }
    }
    return Result::SUCCESS;
}

inline static Result Present(const std::vector<std::shared_ptr<Transform>> transforms) {
    for (const auto& transform : transforms) {
        if (transform->present() != Result::SUCCESS) {
            return Result::ERROR;
        }
    }
    return Result::SUCCESS;
}
};

} // namespace Jetstream

#endif
