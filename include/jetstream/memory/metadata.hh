#ifndef JETSTREAM_MEMORY_METADATA_HH
#define JETSTREAM_MEMORY_METADATA_HH

#include <any>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream {

struct TensorPrototypeMetadata {
    std::vector<U64> shape = {0};
    std::vector<U64> strides = {0};
    U64 type_size = 0;
    U64 hash = 0;
    U64 size = 0;
    U64 size_bytes = 0;
    std::unordered_map<std::string, std::any> store;

    TensorPrototypeMetadata() = default;
    TensorPrototypeMetadata(const TensorPrototypeMetadata&) = delete;
    TensorPrototypeMetadata& operator=(const TensorPrototypeMetadata&) = delete;
};

struct TensorStorageMetadata {
    Device root_device = Device::None;
    std::unordered_set<Device> compatible_devices;
    std::unordered_map<Device, std::any> clones;

    TensorStorageMetadata() = default;
    TensorStorageMetadata(const TensorStorageMetadata&) = delete;
    TensorStorageMetadata& operator=(const TensorStorageMetadata&) = delete;
};

}  // namespace Jetstream

#endif
