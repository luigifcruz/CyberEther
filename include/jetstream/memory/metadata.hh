#ifndef JETSTREAM_MEMORY_METADATA_HH
#define JETSTREAM_MEMORY_METADATA_HH

#include <any>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream {

struct TensorPrototypeMetadata {
    Locale locale;
    std::vector<U64> shape = {0};
    std::vector<U64> stride = {0};
    U64 offset = 0;
    bool contiguous = true;
    U64 element_size = 0;
    U64 hash = 0;

    U64 size = 0;
    U64 size_bytes = 0;
    U64 offset_bytes = 0;
    std::vector<U64> shape_minus_one = {0};
    std::vector<U64> backstride = {0};

    TensorPrototypeMetadata& operator=(const TensorPrototypeMetadata& other);
    TensorPrototypeMetadata& operator=(TensorPrototypeMetadata&& other) noexcept;
};

struct TensorStorageMetadata {
    Device root_device = Device::None;
    std::unordered_set<Device> compatible_devices;
    std::unordered_map<Device, std::any> clones;

    struct Attribute {
     public:
        template<typename T>
        void set(const T& value) {
            if (storage.has_value()) {
                const auto last = std::any_cast<const T&>(storage);
                if (last == value) {
                    return;
                }
            }
            storage = value;
            notify();
        }

        template<typename T>
        const T& get() const {
            return std::any_cast<const T&>(storage);
        }

        constexpr const std::any& get() const {
            return storage;
        }

        constexpr std::any& get() {
            return storage;
        }

        Result subscribe(const Locale& locale, const std::function<void()>& callback);
        Result unsubscribe(const Locale& locale);
        void notify();
    
     private:
        std::any storage;
        std::unordered_map<Locale, std::function<void()>, Locale::Hasher> subscribers = {};
    };

    typedef std::unordered_map<std::string, Attribute> AttributeMap;
    AttributeMap attributes;

    TensorStorageMetadata() = default;
    TensorStorageMetadata(const TensorStorageMetadata&) = delete;
    TensorStorageMetadata& operator=(const TensorStorageMetadata&) = delete;
};

}  // namespace Jetstream

#endif
