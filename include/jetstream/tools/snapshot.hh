#ifndef JETSTREAM_TOOLS_SNAPSHOT_HH
#define JETSTREAM_TOOLS_SNAPSHOT_HH

#include <atomic>
#include <memory>
#include <type_traits>
#include <utility>

namespace Jetstream::Tools {

template<typename T, typename Enable = void>
class Snapshot;

template<typename T>
class Snapshot<T, std::enable_if_t<std::is_trivially_copyable_v<T>>> {
 public:
    Snapshot() : value(T{}) {}
    Snapshot(const T& initial) : value(initial) {}
    Snapshot(T&& initial) : value(std::move(initial)) {}

    Snapshot(const Snapshot&) = delete;
    Snapshot& operator=(const Snapshot&) = delete;
    Snapshot(Snapshot&&) = delete;
    Snapshot& operator=(Snapshot&&) = delete;

    void publish(const T& newValue) {
        value.store(newValue, std::memory_order_release);
    }

    void publish(T&& newValue) {
        value.store(std::move(newValue), std::memory_order_release);
    }

    T get() const {
        return value.load(std::memory_order_acquire);
    }

 private:
    std::atomic<T> value;
};

template<typename T>
class Snapshot<T, std::enable_if_t<!std::is_trivially_copyable_v<T>>> {
 public:
    Snapshot() : value(std::make_shared<const T>()) {}
    Snapshot(const T& initial) : value(std::make_shared<const T>(initial)) {}
    Snapshot(T&& initial) : value(std::make_shared<const T>(std::move(initial))) {}

    Snapshot(const Snapshot&) = delete;
    Snapshot& operator=(const Snapshot&) = delete;
    Snapshot(Snapshot&&) = delete;
    Snapshot& operator=(Snapshot&&) = delete;

    void publish(const T& newValue) {
        std::atomic_store_explicit(&value,
                                   std::make_shared<const T>(newValue),
                                   std::memory_order_release);
    }

    void publish(T&& newValue) {
        std::atomic_store_explicit(&value,
                                   std::make_shared<const T>(std::move(newValue)),
                                   std::memory_order_release);
    }

    T get() const {
        return *std::atomic_load_explicit(&value, std::memory_order_acquire);
    }

 private:
    std::shared_ptr<const T> value;
};

}  // namespace Jetstream::Tools

#endif  // JETSTREAM_TOOLS_SNAPSHOT_HH
