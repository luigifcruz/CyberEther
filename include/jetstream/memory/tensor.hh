#ifndef JETSTREAM_MEMORY_TENSOR_HH
#define JETSTREAM_MEMORY_TENSOR_HH

#include <any>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/memory/buffer.hh"
#include "jetstream/memory/token.hh"
#include "jetstream/memory/types.hh"

namespace Jetstream {

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
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
    Result copyFrom(const Tensor& source);
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
    Result slice(const std::vector<Token>& tokens);
    Result permute(const Shape& axes);

    bool hasAttribute(const std::string& key) const;
    std::vector<std::string> attributeKeys() const;
    Result setAttribute(const std::string& key, const std::any& value);
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
        return Tensor::at<T>(indices...);
    }

    template<typename... Indices>
    const constexpr T& at(Indices... indices) const {
        return Tensor::at<T>(indices...);
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
            s.back() *= sizeof(T);
            return s;
        }
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_MEMORY_TENSOR_HH
