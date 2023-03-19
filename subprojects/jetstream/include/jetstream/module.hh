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
    Result initInput(const T& buffer, const typename T::ShapeType& shape) {
        if (buffer.empty()) {
            JST_DEBUG("Input is empty, allocating.");
            const_cast<T&>(buffer) = std::move(T(shape));
            return Result::SUCCESS;
        }

        if (buffer.shape() != shape) {
            JST_FATAL("Input shape ({}) doesn't match the configuration shape ({}).",
                buffer.shape(), shape);
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    template<typename T>
    Result initOutput(T& buffer, const typename T::ShapeType& shape) {
        if (!buffer.empty()) {
            JST_FATAL("The output buffer should be empty on initialization.");
            return Result::ERROR;
        }

        buffer = std::move(T(shape));

        return Result::SUCCESS;
    }

    friend class Instance;
};

class JETSTREAM_API Compute {
 public:
    virtual ~Compute() = default;

    virtual constexpr const Result createCompute(const RuntimeMetadata& meta) = 0;
    virtual constexpr const Result destroyCompute(const RuntimeMetadata& meta) {
        return Result::SUCCESS;
    }
    virtual constexpr const Result compute(const RuntimeMetadata& meta) = 0;
};

class JETSTREAM_API Present {
 public:
    virtual ~Present() = default;

    virtual constexpr const Result createPresent(Render::Window& window) = 0;
    virtual constexpr const Result destroyPresent(Render::Window& window) {
        return Result::SUCCESS;
    }
    virtual constexpr const Result present(Render::Window& window) = 0;
};

}  // namespace Jetstream

#endif
