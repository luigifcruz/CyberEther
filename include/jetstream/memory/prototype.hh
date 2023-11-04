#ifndef JETSTREAM_MEMORY_PROTOTYPE_HH
#define JETSTREAM_MEMORY_PROTOTYPE_HH

#include <any>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <cassert>

#include "jetstream/types.hh"
#include "jetstream/memory/metadata.hh"

namespace Jetstream {

class TensorPrototype {
 public:
    const U64& size() const noexcept {
        return prototype->size;
    }

    const U64& size_bytes() const noexcept {
        return prototype->size_bytes;
    }

    const U64& type_size() const noexcept {
        return prototype->type_size;
    }

    const U64& hash() const noexcept {
        return prototype->hash;
    }

    const std::vector<U64>& shape() const noexcept {
        return prototype->shape;
    }

    const std::vector<U64>& strides() const noexcept {
        return prototype->strides;
    }

    const U64& shape(const U64& idx) const noexcept {
        return prototype->shape[idx];
    }

    const U64& strides(const U64& idx) const noexcept {
        return prototype->strides[idx];
    }

    bool empty() const noexcept {
        return prototype->size == 0;
    }

    U64 rank() const noexcept {
        return prototype->shape.size();
    }

    U64 ndims() const noexcept {
        return prototype->shape.size();
    }

    const std::unordered_map<std::string, std::any>& store() const noexcept {
        return prototype->store;
    }

    std::unordered_map<std::string, std::any>& store() noexcept {
        return prototype->store;
    }

    template<typename T>
    const T& store(const std::string& key) const noexcept {
        return std::any_cast<const T&>(prototype->store.at(key));
    }

    bool operator==(const TensorPrototype& other) const noexcept {
        return prototype->hash == other.prototype->hash;
    }

    bool operator!=(const TensorPrototype& other) const noexcept {
        return prototype->hash != other.prototype->hash;
    }
    
 protected:
    TensorPrototype() {
        prototype = std::make_shared<TensorPrototypeMetadata>();
    }

    U64 shapeToOffset(const std::vector<U64>& shape) const {
        // TODO: Update for loop to use U64.
        U64 offset = 0;
        for (int i = 0; i < shape.size(); i++) {
            offset += shape[i] * prototype->strides[i];
        }
        return offset;
    }

    // TODO: Add offsetToShape method.

    std::shared_ptr<TensorPrototypeMetadata> prototype;
};

}  // namespace Jetstream

#endif
