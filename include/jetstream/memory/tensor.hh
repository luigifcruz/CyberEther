#ifndef JETSTREAM_MEMORY_TENSOR_HH
#define JETSTREAM_MEMORY_TENSOR_HH

#include <any>
#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/memory/buffer.hh"
#include "jetstream/memory/token.hh"
#include "jetstream/memory/types.hh"
#include "jetstream/tools/numeric.hh"

namespace Jetstream {

class JETSTREAM_API Tensor : public std::enable_shared_from_this<Tensor> {
 public:
    struct SlicePlan {
     private:
        std::shared_ptr<const void> data;

        friend class Tensor;
    };

    Tensor();
    Tensor(void* pointer, const DeviceType& device, const DataType& dtype, const Shape& shape);
    Tensor(const DeviceType& device, const DataType& dtype, const Shape& shape);
    Tensor(const DeviceType& device, const Tensor& source);

    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    ~Tensor();

    Result create(const DeviceType& device, const DataType& dtype, const Shape& shape,
                  const Buffer::Config& config = {});
    Result create(void* pointer, const DeviceType& device, const DataType& dtype, const Shape& shape);
    Result create(const DeviceType& device, const Tensor& source);

    Tensor clone() const;

    bool hasDevice(const DeviceType& device);
    Result copyFrom(const Tensor& source, void* context = nullptr);
    Result swapBuffers(Tensor& other);

    const DeviceType& device() const;
    const DeviceType& nativeDevice() const;
    const DataType& dtype() const;

    const U64& size() const;
    const U64& sizeBytes() const;
    const U64& elementSize() const;

    bool contiguous() const;
    bool empty() const;
    bool validShape() const;

    const U64& offset() const;
    const U64& offsetBytes() const;

    const Shape& shape() const;
    const Shape& stride() const;
    const Shape& shapeMinusOne() const;
    const Shape& backstride() const;

    const U64& shape(Index idx) const;
    const U64& stride(Index idx) const;

    Index rank() const;
    Index ndims() const;

    U64 shapeToOffset(const std::initializer_list<U64>& coordinates) const;

    Result expandDims(Index axis);
    Result squeezeDims(Index axis);
    Result reshape(const Shape& newShape);
    Result broadcastTo(const Shape& newShape);
    Result planSlice(const std::vector<Token>& tokens, SlicePlan& plan) const;
    Result slice(const std::vector<Token>& tokens);
    Result applySlicePlan(const SlicePlan& plan);
    Result permute(const Shape& axes);

    bool hasAttribute(const std::string& key) const;
    std::vector<std::string> attributeKeys() const;
    Result setAttribute(const std::string& key, const std::any& value);
    Result removeAttribute(const std::string& key);
    Result setDerivedAttribute(const std::string& key,
                               std::function<std::any()> compute);
    std::any attribute(const std::string& key) const;
    Result propagateAttributes(const Tensor& source);

    void* data();
    const void* data() const;

    template<typename T>
    T* data() {
        return static_cast<T*>(data());
    }

    template<typename T>
    const T* data() const {
        return static_cast<const T*>(data());
    }

    const Buffer& buffer() const;
    Buffer& buffer();

    const Index& id() const;

    template<typename T, typename... Indices>
    constexpr T& at(Indices... indices) {
        return static_cast<T*>(data())[shapeToOffset({static_cast<U64>(indices)...})];
    }

    template<typename T, typename... Indices>
    const constexpr T& at(Indices... indices) const {
        return static_cast<const T*>(data())[shapeToOffset({static_cast<U64>(indices)...})];
    }

 private:
    struct Impl;
    std::shared_ptr<Impl> impl;

    void ensureImpl();
};

template<typename T>
class TypedTensor : public Tensor {
public:
    using value_type = T;

    using Tensor::Tensor;

    TypedTensor(const DeviceType& device, const Shape& shape)
         : Tensor(device, StorageDataType(), StorageShape(shape)) {}

    Result create(const DeviceType& device, const Shape& shape) {
        return Tensor::create(device, StorageDataType(), StorageShape(shape));
    }

    T* data() {
        return static_cast<T*>(Tensor::data());
    }

    const T* data() const {
        return static_cast<const T*>(Tensor::data());
    }

    template<typename... Indices>
    constexpr T& at(Indices... indices) {
        if constexpr (TypeToDataType<T>() != DataType::None) {
            return Tensor::at<T>(indices...);
        } else {
            const std::array<U64, sizeof...(Indices)> coordinates{
                static_cast<U64>(indices)...
            };
            return *const_cast<T*>(ObjectAddress(coordinates));
        }
    }

    template<typename... Indices>
    const constexpr T& at(Indices... indices) const {
        if constexpr (TypeToDataType<T>() != DataType::None) {
            return Tensor::at<T>(indices...);
        } else {
            const std::array<U64, sizeof...(Indices)> coordinates{
                static_cast<U64>(indices)...
            };
            return *ObjectAddress(coordinates);
        }
    }

    U64 size() const {
        if constexpr (TypeToDataType<T>() != DataType::None) {
            return Tensor::size();
        } else {
            return Tensor::size() / sizeof(T);
        }
    }

 private:
    static constexpr DataType StorageDataType() {
        if constexpr (TypeToDataType<T>() != DataType::None) {
            return TypeToDataType<T>();
        } else {
            return DataType::U8;
        }
    }

    static Shape StorageShape(const Shape& shape) {
        if constexpr (TypeToDataType<T>() != DataType::None) {
            return shape;
        } else {
            Shape s = shape;
            if (s.empty()) {
                return s;
            }

            U64 storageExtent = 0;
            if (!detail::CheckedMultiply(s.back(), static_cast<U64>(sizeof(T)),
                                         storageExtent)) {
                return {};
            }
            s.back() = storageExtent;
            return s;
        }
    }

    template<std::size_t Rank>
    const T* ObjectAddress(const std::array<U64, Rank>& coordinates) const {
        if (!validShape() || dtype() != DataType::U8 || elementSize() != 1) {
            throw std::logic_error("Custom TypedTensor has an incompatible storage layout");
        }
        if (device() != DeviceType::CPU) {
            throw std::logic_error("Custom TypedTensor element access requires CPU storage");
        }

        const auto& storageShape = shape();
        const auto& storageStride = stride();
        if (storageShape.empty() || storageShape.size() != storageStride.size()) {
            throw std::logic_error("Custom TypedTensor has an invalid storage layout");
        }
        if (Rank != storageShape.size()) {
            throw std::out_of_range("Custom TypedTensor coordinate rank does not match tensor rank");
        }

        constexpr U64 objectSize = static_cast<U64>(sizeof(T));
        if (storageStride.back() != 1 ||
            storageShape.back() % objectSize != 0 ||
            offsetBytes() % objectSize != 0) {
            throw std::logic_error("Custom TypedTensor packed byte axis is invalid");
        }

        U64 byteOffset = 0;
        for (std::size_t i = 0; i < Rank; ++i) {
            U64 storageCoordinate = coordinates[i];
            U64 logicalExtent = storageShape[i];
            if (i + 1 == Rank) {
                logicalExtent /= objectSize;
            }
            if (storageCoordinate >= logicalExtent) {
                throw std::out_of_range("Custom TypedTensor coordinate is outside the tensor shape");
            }

            if (i + 1 == Rank &&
                !detail::CheckedMultiply(storageCoordinate, objectSize, storageCoordinate)) {
                throw std::overflow_error("Custom TypedTensor coordinate exceeds byte storage");
            }

            U64 term = 0;
            if (!detail::CheckedMultiply(storageCoordinate, storageStride[i], term) ||
                !detail::CheckedAdd(byteOffset, term, byteOffset)) {
                throw std::overflow_error("Custom TypedTensor byte offset overflow");
            }
        }

        U64 absoluteOffset = 0;
        U64 objectEnd = 0;
        if (!detail::CheckedAdd(offsetBytes(), byteOffset, absoluteOffset) ||
            !detail::CheckedAdd(absoluteOffset, objectSize, objectEnd)) {
            throw std::overflow_error("Custom TypedTensor storage range overflow");
        }

        const auto& activeBuffer = buffer();
        if (!activeBuffer.valid() || activeBuffer.device() != DeviceType::CPU) {
            throw std::logic_error("Custom TypedTensor does not have a CPU backing buffer");
        }
        if (objectEnd > activeBuffer.sizeBytes()) {
            throw std::out_of_range("Custom TypedTensor object exceeds its backing buffer");
        }

        const auto* bytes = static_cast<const std::uint8_t*>(activeBuffer.data());
        if (!bytes) {
            throw std::out_of_range("Custom TypedTensor has no accessible storage");
        }

        const auto address = reinterpret_cast<std::uintptr_t>(bytes);
        if (objectEnd > std::numeric_limits<std::uintptr_t>::max() - address) {
            throw std::overflow_error("Custom TypedTensor pointer range overflow");
        }
        if ((address + static_cast<std::uintptr_t>(absoluteOffset)) % alignof(T) != 0) {
            throw std::runtime_error("Custom TypedTensor object address is misaligned");
        }

        return reinterpret_cast<const T*>(bytes + absoluteOffset);
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_MEMORY_TENSOR_HH
