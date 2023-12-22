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
    std::vector<U64> strides = {0};
    U64 type_size = 0;
    U64 hash = 0;
    U64 size = 0;
    U64 size_bytes = 0;

    TensorPrototypeMetadata& operator=(const TensorPrototypeMetadata& other) {
        if (locale.empty()) {
            locale = other.locale;
        }
        shape = other.shape;
        strides = other.strides;
        type_size = other.type_size;
        hash = other.hash;
        size = other.size;
        size_bytes = other.size_bytes;
        return *this;
    }

    TensorPrototypeMetadata& operator=(TensorPrototypeMetadata&& other) noexcept {
        if (locale.empty()) {
            locale = std::move(other.locale);
        }
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        type_size = std::move(other.type_size);
        hash = std::move(other.hash);
        size = std::move(other.size);
        size_bytes = std::move(other.size_bytes);
        return *this;
    }
};

struct TensorStorageMetadata {
    Device root_device = Device::None;
    std::unordered_set<Device> compatible_devices;
    std::unordered_map<Device, std::any> clones;

    struct Attribute {
     public:
        template<typename T>
        void set(const T& value) {
            storage = value;
            notify();
        }

        template<typename T>
        const T& get() const {
            return std::any_cast<const T&>(storage);
        }

        const std::any& get() const {
            return storage;
        }

        std::any& get() {
            return storage;
        }

        Result subscribe(const Locale& locale, const std::function<void()>& callback) {
            if (subscribers.contains(locale)) {
                JST_ERROR("[METADATA] Attribute already has a subscriber for locale {}.", locale);
                return Result::ERROR;
            }
            subscribers[locale] = callback;
            return Result::SUCCESS;
        }

        Result unsubscribe(const Locale& locale) {
            if (!subscribers.contains(locale)) {
                return Result::SUCCESS;
            }
            subscribers.erase(locale);
            return Result::SUCCESS;
        }

        void notify() {
            for (const auto& [locale, callback] : subscribers) {
                callback();
            }
        }
    
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
