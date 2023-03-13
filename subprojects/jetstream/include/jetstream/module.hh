#ifndef JETSTREAM_MODULE_HH
#define JETSTREAM_MODULE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/metadata.hh"
#include "jetstream/render/base.hh"

namespace Jetstream {

class JETSTREAM_API Module {
 public:
    virtual ~Module() = default;

    virtual void summary() const = 0;

 protected:
    virtual constexpr const Device device() const = 0;
    virtual constexpr const Taint taints() const = 0;
    
    template<typename T>
    Result initInput(const T& buffer, const U64& size) {
        if (buffer.empty()) {
            JST_DEBUG("Input is empty, allocating {} elements", size);
            const_cast<T&>(buffer) = std::move(T(size));
            return Result::SUCCESS;
        }

        if (buffer.size() != size) {
            JST_FATAL("Input size ({}) doesn't match the configuration size ({}).",
                buffer.size(), size);
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    template<typename T>
    Result initOutput(T& buffer, const U64& size) {
        if (!buffer.empty()) {
            JST_FATAL("The output buffer should be empty on initialization.");
            return Result::ERROR;
        }

        buffer = std::move(T(size));

        return Result::SUCCESS;
    }

    friend class Instance;
};

class JETSTREAM_API Compute {
 public:
    virtual ~Compute() = default;

    virtual constexpr const Result createCompute(const RuntimeMetadata& meta) = 0;
    virtual constexpr const Result compute(const RuntimeMetadata& meta) = 0;
};

class JETSTREAM_API Present {
 public:
    virtual ~Present() = default;

    virtual constexpr const Result createPresent(Render::Window& window) = 0;
    virtual constexpr const Result present(Render::Window& window) = 0;
};

}  // namespace Jetstream

#endif
