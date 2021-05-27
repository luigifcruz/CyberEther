#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include "type.hpp"

namespace Jetstream {

class Module {
public:
    explicit Module() {};
    virtual ~Module() = default;

    Result compute(std::shared_ptr<Module> input = nullptr, bool async = true) {
        auto mode = (async) ? std::launch::async : std::launch::deferred;
        future = std::async(mode, [&](){
            if (input) {
                auto result = input->barrier();
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
        return future.get();
    }

    Result present() {
        std::scoped_lock<std::mutex> guard(mutex);
        return this->underlyingPresent();
    }

protected:
    std::mutex mutex;
    std::future<Result> future;

    virtual Result underlyingCompute() = 0;
    virtual Result underlyingPresent() = 0;
};

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
