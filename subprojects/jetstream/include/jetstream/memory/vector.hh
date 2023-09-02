#ifndef JETSTREAM_MEMORY_VECTOR_HH
#define JETSTREAM_MEMORY_VECTOR_HH

#include <memory>
#include <functional>
#include <unordered_map>

#include "jetstream/types.hh"

namespace Jetstream {

template<Device DeviceId, typename DataType, U64 Dimensions = 1> class Vector;

template<U64 Dimensions>
struct VectorShape {
 public:
    std::array<U64, Dimensions> _shape;

    constexpr U64& operator[](const U64& idx) {
        return _shape[idx];
    }

    constexpr const U64& operator[](const U64& idx) const {
        return _shape[idx];
    }

    constexpr auto begin() {
        return _shape.begin();
    }

    constexpr auto end() {
        return _shape.end();
    }

    constexpr auto begin() const {
        return _shape.begin();
    }

    constexpr auto end() const {
        return _shape.end();
    }

    constexpr auto empty() const {
        bool empty = true;
        for (const auto& dim : _shape) {
            empty &= (dim == 0);
        }
        return empty;
    }

    constexpr std::vector<U64> native() const noexcept {
        return std::vector<U64>(_shape.begin(), _shape.end());
    }
};

class VectorType {};

template<typename Type, U64 Dimensions>
class VectorImpl : public VectorType {
 public:
    using DataType = Type;

    virtual ~VectorImpl() {
        decreaseRefCount();
    }

    constexpr DataType* data() const noexcept {
        return _data;
    }

    constexpr const VectorShape<Dimensions>& shape() const noexcept {
        return _shape;
    }

    constexpr U64 size() const noexcept {
        U64 _size = 1;
        for (const auto& dim : _shape) {
            _size *= dim;
        }
        return _size;
    }

    constexpr U64 refs() const noexcept {
        if (!_refs) {
            return 0;
        }
        return *_refs;
    }

    constexpr U64 hash() const noexcept {
        if (!_data) {
            return 0;
        }
        return HashU64(reinterpret_cast<U64>(_data));
    }

    constexpr Locale locale() const noexcept {
        return _locale;
    }

    constexpr U64 size_bytes() const noexcept {
        return size() * sizeof(DataType);
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return (_data == nullptr);
    }

    U64 shapeToOffset(const VectorShape<Dimensions>& shape) const {
        U64 offset = 0;
        for (U64 i = 0; i < Dimensions; i++) {
            U64 product = shape[i];
            for (U64 j = i + 1; j < Dimensions; j++) {
                product *= _shape[j];
            }
            offset += product;
        }
        return offset;
    }

    constexpr DataType& operator[](const VectorShape<Dimensions>& shape) {
        return _data[shapeToOffset(shape)];
    }

    constexpr const DataType& operator[](const VectorShape<Dimensions>& shape) const {
        return _data[shapeToOffset(shape)];
    }

    constexpr DataType& operator[](const U64& idx) {
        return _data[idx];
    }

    constexpr const DataType& operator[](const U64& idx) const {
        return _data[idx];
    }

    constexpr auto begin() {
        return _data;
    }

    constexpr auto end() {
        return _data + size();
    }

    constexpr auto begin() const {
        return _data;
    }

    constexpr auto end() const {
        return _data + size();
    }

    VectorImpl(VectorImpl&&) = delete;
    VectorImpl& operator=(VectorImpl&&) = delete;

 protected:
    VectorShape<Dimensions> _shape;
    Locale _locale;
    DataType* _data;
    U64* _refs;
    std::vector<std::function<void()>>* _destructors;

    VectorImpl()
             : _shape({0}),
               _locale({}),
               _data(nullptr),
               _refs(nullptr),
               _destructors(nullptr) {
        JST_TRACE("Empty vector created.");
    }

    explicit VectorImpl(const VectorImpl& other)
             : _shape(other._shape),
               _locale(other._locale),
               _data(other._data),
               _refs(other._refs),
               _destructors(other._destructors) {
        JST_TRACE("Vector created by copy.");

        increaseRefCount();
    }

    VectorImpl& operator=(const VectorImpl& other) {
        JST_TRACE("Vector copied to existing.");

        decreaseRefCount();

        _shape = other._shape;
        _data = other._data;
        _locale = other._locale;
        _refs = other._refs;
        _destructors = other._destructors;

        increaseRefCount();

        return *this;
    }

    explicit VectorImpl(void* ptr, const VectorShape<Dimensions>& shape)
             : _shape(shape),
               _locale({}),
               _data(static_cast<DataType*>(ptr)),
               _refs(nullptr),
               _destructors(nullptr) {
        increaseRefCount();
    }

    void decreaseRefCount() {
        if (!_refs) {
            return;
        }
        JST_TRACE("Decreasing reference counter to {}.", *_refs - 1);

        if (--(*_refs) == 0) {
            JST_TRACE("Deleting {} pointers.", _destructors->size());

            for (auto& destructor : *_destructors) {
                destructor();
            }

            delete _destructors;

            reset();
        }
    }

    void increaseRefCount() {
        if (!_refs) {
            JST_TRACE("Creating new destructor list.");
            _destructors = new std::vector<std::function<void()>>();
            return;
        }
        JST_TRACE("Increasing reference counter to {}.", *_refs + 1);
        *_refs += 1;
    }

    void updateLocale(const Locale& locale) {
        JST_TRACE("Updating locale to '{}'.", locale);
        _locale = locale;
    }

    void reset() {
        _shape = {0};
        _locale = {};
        _data = nullptr;
        _refs = nullptr;
        _destructors = nullptr;
    }

    friend class Module;
    friend class Bundle;
};

template <typename T>
struct IsVector : std::false_type {};

template <Device DeviceId, typename DataType, U64 Dimensions>
struct IsVector<Vector<DeviceId, DataType, Dimensions>> : std::true_type {};

}  // namespace Jetstream

#endif
