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
    struct IoMetadata {
        U64 hash;
        void* ptr;
        Device device;
        std::string type;
        std::vector<U64> shape;
    };

    virtual ~Module() = default;

    virtual const std::string name() const = 0;
    virtual void summary() const = 0;

    constexpr const U64& id() const {
        return _id;
    }

 protected:
    virtual constexpr Device device() const = 0;
    
    template<typename T>
    Result initInput(const T& buffer) {
        if (buffer.empty()) {
            JST_FATAL("Module input can't be empty.");
            return Result::ERROR;
        }

        inputs.push_back({
            buffer.hash(),
            buffer.data(),
            buffer.device(),
            NumericTypeInfo<typename T::DataType>::name,
            buffer.shapeVector(),
        });

        return Result::SUCCESS;
    }

    template<typename T>
    Result initOutput(T& buffer, const typename T::ShapeType& shape) {
        if (!buffer.empty()) {
            JST_FATAL("The output buffer should be empty on initialization.");
            return Result::ERROR;
        }

        buffer = T(shape);

        outputs.push_back({
            buffer.hash(),
            buffer.data(),
            buffer.device(),
            NumericTypeInfo<typename T::DataType>::name,
            buffer.shapeVector(),
        });

        return Result::SUCCESS;
    }

    const std::vector<IoMetadata>& getInputs() const {
        return inputs;
    }

    const std::vector<IoMetadata>& getOutputs() const {
        return outputs;
    }

    void setId(const U64& id) {
        _id = id;
    }

 private:
    U64 _id;
    std::vector<IoMetadata> inputs;
    std::vector<IoMetadata> outputs;

    friend class Instance;
};

class JETSTREAM_API Compute {
 public:
    virtual ~Compute() = default;

    virtual constexpr Result createCompute(const RuntimeMetadata& meta) = 0;
    virtual constexpr Result destroyCompute(const RuntimeMetadata&) {
        return Result::SUCCESS;
    }
    virtual constexpr Result compute(const RuntimeMetadata& meta) = 0;
};

class JETSTREAM_API Present {
 public:
    virtual ~Present() = default;

    virtual constexpr Result createPresent(Render::Window& window) = 0;
    virtual constexpr Result destroyPresent(Render::Window&) {
        return Result::SUCCESS;
    }
    virtual constexpr Result present(Render::Window& window) = 0;
};

}  // namespace Jetstream

#endif
