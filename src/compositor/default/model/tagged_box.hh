#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_TAGGED_BOX_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_TAGGED_BOX_HH

#include "jetstream/types.hh"

#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace Jetstream {

namespace Detail {

template<typename T, typename... Ts>
consteval U64 IndexOf() {
    constexpr bool matches[] = {std::is_same_v<T, Ts>...};
    for (U64 index = 0; index < sizeof...(Ts); ++index) {
        if (matches[index]) {
            return index;
        }
    }
    return sizeof...(Ts);
}

template<typename... Ts>
consteval bool AllUnique() {
    constexpr U64 indices[] = {IndexOf<Ts, Ts...>()...};
    for (U64 index = 0; index < sizeof...(Ts); ++index) {
        if (indices[index] != index) {
            return false;
        }
    }
    return true;
}

}  // namespace Detail

template<U64 InlineBytes, typename... Ts>
class TaggedBox {
 public:
    static constexpr U64 Count = sizeof...(Ts);
    static constexpr U16 EmptyIndex = std::numeric_limits<U16>::max();

    static_assert(Count > 0, "TaggedBox needs at least one type");
    static_assert(Count < EmptyIndex, "TaggedBox holds too many types");
    static_assert(Detail::AllUnique<Ts...>(), "TaggedBox types must be unique");
    static_assert(InlineBytes >= sizeof(void*), "TaggedBox inline storage must fit a pointer");

    template<typename T>
    static constexpr bool Registered = Detail::IndexOf<T, Ts...>() < Count;

    template<typename T>
        requires Registered<T>
    static constexpr U16 IndexOf = static_cast<U16>(Detail::IndexOf<T, Ts...>());

    template<typename T>
        requires Registered<T>
    static constexpr bool IsInline = sizeof(T) <= InlineBytes &&
                                     alignof(T) <= alignof(std::max_align_t) &&
                                     std::is_nothrow_move_constructible_v<T>;

    template<template<typename...> typename F>
    using Expand = F<Ts...>;

    TaggedBox() noexcept = default;

    template<typename T>
        requires Registered<std::decay_t<T>>
    TaggedBox(T&& value) noexcept(IsInline<std::decay_t<T>> &&
                                  std::is_nothrow_constructible_v<std::decay_t<T>, T&&>) {
        construct<std::decay_t<T>>(std::forward<T>(value));
        ops = &OpsFor<std::decay_t<T>>;
    }

    TaggedBox(const TaggedBox& other) {
        if (other.ops != nullptr) {
            other.ops->copy(*this, other);
            ops = other.ops;
        }
    }

    TaggedBox(TaggedBox&& other) noexcept {
        take(other);
    }

    TaggedBox& operator=(const TaggedBox& other) {
        if (this != &other) {
            TaggedBox copy(other);
            reset();
            take(copy);
        }
        return *this;
    }

    TaggedBox& operator=(TaggedBox&& other) noexcept {
        if (this != &other) {
            reset();
            take(other);
        }
        return *this;
    }

    ~TaggedBox() {
        reset();
    }

    bool empty() const noexcept {
        return ops == nullptr;
    }

    U16 index() const noexcept {
        return ops != nullptr ? ops->index : EmptyIndex;
    }

    template<typename T>
        requires Registered<T>
    bool is() const noexcept {
        return ops != nullptr && ops->index == IndexOf<T>;
    }

    template<typename... Us>
        requires(sizeof...(Us) > 0 && (Registered<Us> && ...))
    bool isAnyOf() const noexcept {
        if (ops == nullptr) {
            return false;
        }
        constexpr U16 indices[] = {IndexOf<Us>...};
        for (const U16 candidate : indices) {
            if (candidate == ops->index) {
                return true;
            }
        }
        return false;
    }

    template<typename T>
        requires Registered<T>
    T* getIf() noexcept {
        return is<T>() ? static_cast<T*>(ops->data(*this)) : nullptr;
    }

    template<typename T>
        requires Registered<T>
    const T* getIf() const noexcept {
        return is<T>() ? static_cast<const T*>(ops->data(*this)) : nullptr;
    }

    template<typename T>
        requires Registered<T>
    T& get() {
        if (T* value = getIf<T>()) {
            return *value;
        }
        throw std::logic_error("TaggedBox holds a different type");
    }

    template<typename T>
        requires Registered<T>
    const T& get() const {
        if (const T* value = getIf<T>()) {
            return *value;
        }
        throw std::logic_error("TaggedBox holds a different type");
    }

 private:
    struct Ops {
        U16 index;
        void* (*data)(const TaggedBox& box) noexcept;
        void (*copy)(TaggedBox& target, const TaggedBox& source);
        void (*move)(TaggedBox& target, TaggedBox& source) noexcept;
        void (*destroy)(TaggedBox& box) noexcept;
    };

    alignas(std::max_align_t) U8 storage[InlineBytes];
    const Ops* ops = nullptr;

    template<typename T>
    static T* Pointer(const TaggedBox& box) noexcept {
        U8* bytes = const_cast<U8*>(box.storage);
        if constexpr (IsInline<T>) {
            return std::launder(reinterpret_cast<T*>(bytes));
        } else {
            return *std::launder(reinterpret_cast<T**>(bytes));
        }
    }

    template<typename T>
    static void* Data(const TaggedBox& box) noexcept {
        return Pointer<T>(box);
    }

    template<typename T>
    static void Copy(TaggedBox& target, const TaggedBox& source) {
        target.construct<T>(std::as_const(*Pointer<T>(source)));
    }

    template<typename T>
    static void Move(TaggedBox& target, TaggedBox& source) noexcept {
        if constexpr (IsInline<T>) {
            T* value = Pointer<T>(source);
            ::new (static_cast<void*>(target.storage)) T(std::move(*value));
            value->~T();
        } else {
            ::new (static_cast<void*>(target.storage)) T*(Pointer<T>(source));
        }
    }

    template<typename T>
    static void Destroy(TaggedBox& box) noexcept {
        if constexpr (IsInline<T>) {
            Pointer<T>(box)->~T();
        } else {
            delete Pointer<T>(box);
        }
    }

    template<typename T>
    static constexpr Ops OpsFor = {IndexOf<T>, &Data<T>, &Copy<T>, &Move<T>, &Destroy<T>};

    template<typename T, typename Arg>
    void construct(Arg&& arg) {
        if constexpr (IsInline<T>) {
            ::new (static_cast<void*>(storage)) T(std::forward<Arg>(arg));
        } else {
            ::new (static_cast<void*>(storage)) T*(new T(std::forward<Arg>(arg)));
        }
    }

    void take(TaggedBox& other) noexcept {
        if (other.ops != nullptr) {
            other.ops->move(*this, other);
            ops = std::exchange(other.ops, nullptr);
        }
    }

    void reset() noexcept {
        if (ops != nullptr) {
            std::exchange(ops, nullptr)->destroy(*this);
        }
    }
};

}  // namespace Jetstream

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_TAGGED_BOX_HH
