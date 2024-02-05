#ifndef JETSTREAM_MEMORY_UTILS_JUGGLER_H
#define JETSTREAM_MEMORY_UTILS_JUGGLER_H

#include <memory>
#include <vector>

#include "jetstream/types.hh"

namespace Jetstream::Memory {

/**
 * @class Juggler
 * @brief A class that manages a pool of shared pointers to objects of type T.
 * 
 * The Juggler class provides a way to reuse memory by maintaining a pool of shared pointers.
 * It allows objects to be allocated and deallocated efficiently, reducing the overhead of memory allocation.
 * The class provides methods to resize the pool, clear the pool, and retrieve a shared pointer from the pool.
 * Unused pointers are recycled, and unique pointers are added to the used list.
 * 
 * @tparam T The type of objects managed by the Juggler.
 */
template<typename T>
class Juggler {
 public:
    /**
     * @brief Default constructor.
     */
    Juggler() = default;

    /**
     * @brief Constructor that resizes the pool and initializes objects.
     * 
     * @tparam Args Variadic template parameter pack for object initialization arguments.
     * @param size The size of the pool.
     * @param args The arguments to initialize the objects.
     */
    template<typename... Args>
    Juggler(const U64& size, Args&&... args) {
        resize(size, std::forward<Args>(args)...);
    }

    /**
     * @brief Resizes the pool and initializes objects.
     * 
     * @tparam Args Variadic template parameter pack for object initialization arguments.
     * @param size The new size of the pool.
     * @param args The arguments to initialize the objects.
     */
    template<typename... Args>
    void resize(const U64& size, Args&&... args) {
        clear();
        pool.reserve(size);
        used.reserve(size);
        for (U64 i = 0; i < size; ++i) {
            pool.push_back(std::make_shared<T>(std::forward<Args>(args)...));
        }
    }

    /**
     * @brief Clears the pool and used objects.
     */
    void clear() {
        pool.clear();
        used.clear();
    }

    /**
     * @brief Retrieves a reusable object from the pool.
     * 
     * @return A shared pointer to the retrieved object, or nullptr if the pool is empty.
     */
    std::shared_ptr<T> get() {
        // Recycle unused pointers.

        for (auto it = used.begin(); it != used.end();) {
            if ((*it).unique()) {
                pool.push_back(*it);
                it = used.erase(it);
            } else {
                ++it;
            }
        }

        // Check if there are any pointers available.

        if (pool.empty()) {
            return nullptr;
        }

        // Get the pointer from the pool.

        auto ptr = pool.back();
        pool.pop_back();

        // Add the pointer to the used list.

        used.push_back(ptr);

        // Return the pointer to caller.

        return ptr;
    }

 private:
    std::vector<std::shared_ptr<T>> pool;
    std::vector<std::shared_ptr<T>> used;
};

}  // namespace Jetstream::Memory

#endif

