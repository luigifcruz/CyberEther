#ifndef JETSTREAM_MODULE_HH
#define JETSTREAM_MODULE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/metadata.hh"
#include "jetstream/render/base.hh"
#include "jetstream/memory/base.hh"

namespace Jetstream {

class JETSTREAM_API Module {
 public:
    struct IoMetadata {
        U64 hash;
        U64 phash;
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
            buffer.phash(),
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
            buffer.phash(),
            buffer.data(),
            buffer.device(),
            NumericTypeInfo<typename T::DataType>::name,
            buffer.shapeVector(),
        });

        return Result::SUCCESS;
    }

    template<Device DeviceId, typename Type, U64 Dimensions>
    Result initInplaceOutput(Vector<DeviceId, Type, Dimensions>& dst,
                             const Vector<DeviceId, Type, Dimensions>& src) {
        dst = const_cast<Vector<DeviceId, Type, Dimensions>&>(src);

        dst.incrementPositionalIndex();

        outputs.push_back({
            dst.hash(),
            dst.phash(),
            dst.data(),
            dst.device(),
            NumericTypeInfo<Type>::name,
            dst.shapeVector(),
        });

        return Result::SUCCESS;
    }

    // TODO: Add initInplaceOutput with reshape.

    const std::vector<IoMetadata>& getInputs() const {
        return inputs;
    }

    const std::vector<IoMetadata>& getOutputs() const {
        return outputs;
    }

    void setId(const U64& id) {
        _id = id;
    }

    template<typename T>
    static Result BindVariable(std::unordered_map<std::string, std::any>& map,
                               const std::string& name,
                               const T& variable) {
        if (map.count(name) == 0) {
            JST_FATAL("Varible name not found inside map.");
            return Result::ERROR;
        }

        auto& anyVar = map[name];

        if (!anyVar.has_value()) {
            JST_ERROR("Variable not initialized.");
            return Result::ERROR;
        }

        try {
            const_cast<T&>(variable) = std::any_cast<T>(anyVar);
        } catch (const std::bad_any_cast& e) {
            JST_ERROR("Failed to cast from any.");
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    template<typename T>
    static Result RegisterVariable(std::unordered_map<std::string, std::any>& map,
                                   const std::string& name,
                                   T& variable) {
        map[name] = std::any(variable);

        return Result::SUCCESS;
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
