#include "jetstream/memory/metadata.hh" 

namespace Jetstream {

TensorPrototypeMetadata& TensorPrototypeMetadata::operator=(const TensorPrototypeMetadata& other) {
    if (locale.empty()) {
        locale = other.locale;
    }
    shape = other.shape;
    stride = other.stride;
    element_size = other.element_size;
    offset = other.offset;
    contiguous = other.contiguous;
    hash = other.hash;

    size = other.size;
    size_bytes = other.size_bytes;
    offset_bytes = other.offset_bytes;
    shape_minus_one = other.shape_minus_one;
    backstride = other.backstride;
    return *this;
}

TensorPrototypeMetadata& TensorPrototypeMetadata::operator=(TensorPrototypeMetadata&& other) noexcept {
    if (locale.empty()) {
        locale = std::move(other.locale);
    }
    shape = std::move(other.shape);
    stride = std::move(other.stride);
    element_size = std::move(other.element_size);
    offset = std::move(other.offset);
    contiguous = std::move(other.contiguous);
    hash = std::move(other.hash);

    size = std::move(other.size);
    size_bytes = std::move(other.size_bytes);
    offset_bytes = std::move(other.offset_bytes);
    shape_minus_one = std::move(other.shape_minus_one);
    backstride = std::move(other.backstride);
    return *this;
}

Result TensorStorageMetadata::Attribute::subscribe(const Locale& locale, const std::function<void()>& callback) {
    if (subscribers.contains(locale)) {
        JST_ERROR("[METADATA] Attribute already has a subscriber for locale {}.", locale);
        return Result::ERROR;
    }
    subscribers[locale] = callback;
    return Result::SUCCESS;
}

Result TensorStorageMetadata::Attribute::unsubscribe(const Locale& locale) {
    if (!subscribers.contains(locale)) {
        return Result::SUCCESS;
    }
    subscribers.erase(locale);
    return Result::SUCCESS;
}

void TensorStorageMetadata::Attribute::notify() {
    for (const auto& [locale, callback] : subscribers) {
        callback();
    }
}

}  // namespace Jetstream