#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include <complex>
#include <future>
#include <iostream>
#include <vector>

#include "jetstream_config.hpp"

#ifndef JETSTREAM_ASSERT_SUCCESS
#define JETSTREAM_ASSERT_SUCCESS(result) { \
    if (result != Jetstream::Result::SUCCESS) { \
        std::cerr << "Jetstream encountered an exception (" <<  magic_enum::enum_name(result) << ") in " \
            << __PRETTY_FUNCTION__ << " in line " << __LINE__ << " of file " << __FILE__ << "." << std::endl; \
        throw result; \
    } \
}
#endif

namespace Jetstream {

enum Result {
    SUCCESS = 0,
    ERROR = 1,
    ERROR_FUTURE_INVALID,
};

namespace DF {
    namespace CPU {
        typedef struct CF32V {
            std::shared_ptr<std::vector<std::complex<float>>> input;
            std::shared_ptr<std::vector<std::complex<float>>> output;
        } CF32V;
    }

    namespace CUDA {
        typedef struct CF32V {
            float* input = nullptr;
            size_t input_size;
            float* output = nullptr;
            size_t output_size;
        } CF32V;
    }
}

class Transform {
public:
    explicit Transform() {};
    virtual ~Transform() = default;

    Result compute(std::shared_ptr<Transform> input = nullptr, bool async = true) {
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

inline Result Barrier(const std::vector<std::shared_ptr<Transform>> transforms) {
    for (const auto& transform : transforms) {
        auto result = transform->barrier();
        if (result != Result::SUCCESS) {
            return result;
        }
    }
    return Result::SUCCESS;
}

inline Result Present(const std::vector<std::shared_ptr<Transform>> transforms) {
    for (const auto& transform : transforms) {
        auto result = transform->present();
        if (result != Result::SUCCESS) {
            return result;
        }
    }
    return Result::SUCCESS;
}

} // namespace Jetstream

#endif
