#ifndef JETSTREAM_MODULE_HH
#define JETSTREAM_MODULE_HH

#include <string>
#include <algorithm>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/helpers.hh"
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
    
    template<Device DeviceId, typename DataType, U64 Dimensions>
    Result initInput(const Vector<DeviceId, DataType, Dimensions>& buffer) {
        if (buffer.empty()) {
            JST_FATAL("Module input can't be empty.");
            return Result::ERROR;
        }

        inputs.push_back({
            buffer.hash(),
            buffer.phash(),
            buffer.data(),
            buffer.device(),
            NumericTypeInfo<DataType>::name,
            buffer.shape().native(),
        });

        return Result::SUCCESS;
    }

    template<Device DeviceId, typename DataType, U64 Dimensions>
    Result initOutput(Vector<DeviceId, DataType, Dimensions>& buffer,
                      const VectorShape<Dimensions>& shape) {
        if (!buffer.empty()) {
            JST_FATAL("The output buffer should be empty on initialization.");
            return Result::ERROR;
        }

        buffer = Vector<DeviceId, DataType, Dimensions>(shape);

        outputs.push_back({
            buffer.hash(),
            buffer.phash(),
            buffer.data(),
            buffer.device(),
            NumericTypeInfo<DataType>::name,
            buffer.shape().native(),
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
            dst.shape().native(),
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
                               const T& variable,
                               const bool& castFromString = false) {
        if (map.count(name) == 0) {
            JST_FATAL("Varible name ({}) not found inside map.", name);
            return Result::ERROR;
        }

        auto& anyVar = map[name];

        if (!anyVar.has_value()) {
            JST_ERROR("Variable ({}) not initialized.", name);
            return Result::ERROR;
        }

        try {
            if (!castFromString) {
                const_cast<T&>(variable) = std::any_cast<T>(anyVar);
            } else if constexpr (std::is_same<T, std::string>::value) {
                JST_TRACE("BindVariable: Converting std::string to std::string.");
                const_cast<T&>(variable) = std::any_cast<std::string>(anyVar);
            } else if constexpr (std::is_same<T, U64>::value) {
                JST_TRACE("BindVariable: Converting std::string to U64.");
                const_cast<T&>(variable) = std::stoull(std::any_cast<std::string>(anyVar));
            } else if constexpr (std::is_same<T, F32>::value) {
                JST_TRACE("BindVariable: Converting std::string to F32.");
                const_cast<T&>(variable) = std::stof(std::any_cast<std::string>(anyVar));
            } else if constexpr (std::is_same<T, F64>::value) {
                JST_TRACE("BindVariable: Converting std::string to F64.");
                const_cast<T&>(variable) = std::stod(std::any_cast<std::string>(anyVar));
            } else if constexpr (std::is_same<T, bool>::value) {
                JST_TRACE("BindVariable: Converting std::string to BOOL.");
                std::string lower_s = std::any_cast<std::string>(anyVar);
                std::transform(lower_s.begin(), lower_s.end(), lower_s.begin(), ::tolower);
                const_cast<T&>(variable) = lower_s == "true" || lower_s == "1";
            } else if constexpr (std::is_same<T, VectorShape<2>>::value) {
                JST_TRACE("BindVariable: Converting std::string to VectorShape<2>.");
                const auto values = Helper::SplitString(std::any_cast<std::string>(anyVar), ", ");
                JST_ASSERT_THROW(values.size() == 2);
                const_cast<T&>(variable) = VectorShape<2>{std::stoull(values[0]), std::stoull(values[1])};
            } else if constexpr (std::is_same<T, Range<F32>>::value) {
                JST_TRACE("BindVariable: Converting std::string to Range<F32>.");
                const auto values = Helper::SplitString(std::any_cast<std::string>(anyVar), ", ");
                JST_ASSERT_THROW(values.size() == 2);
                const_cast<T&>(variable) = Range<F32>{std::stof(values[0]), std::stof(values[1])};
            } else if constexpr (std::is_same<T, Render::Size2D<U64>>::value) {
                JST_TRACE("BindVariable: Converting std::string to BOOL.");
                const auto values = Helper::SplitString(std::any_cast<std::string>(anyVar), ", ");
                JST_ASSERT_THROW(values.size() == 2);
                const_cast<T&>(variable) = Render::Size2D<U64>{std::stoull(values[0]), std::stoull(values[1])};
            } else {
                JST_FATAL("Can't find cast operator for variable ({}).", name);
                JST_CHECK_THROW(Result::ERROR);
            }
        } catch (const std::bad_any_cast& e) {
            JST_ERROR("Variable ({}) failed to cast from any: {}", name, e.what());
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
