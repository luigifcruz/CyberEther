#ifndef JETSTREAM_MODULE_HH
#define JETSTREAM_MODULE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/metadata.hh"

namespace Jetstream {

class JETSTREAM_API Module {
 public:
    virtual ~Module() = default;

 protected:
    virtual constexpr const Result compute(const RuntimeMetadata& meta = {}) {
        return Result::SUCCESS;
    }

    virtual constexpr const Result present(const RuntimeMetadata& meta = {}) {
        return Result::SUCCESS;
    }

    template<typename T>
    static Result InitInput(const T& buffer, std::size_t size) {
        if (buffer.empty()) {
            JST_DEBUG("Input is empty, allocating {} elements", size);
            return const_cast<T&>(buffer).resize(size);
        }

        if (buffer.size() != size) {
            JST_FATAL("Input size ({}) doesn't match the configuration size ({}).",
                buffer.size(), size);
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    template<typename T>
    static Result InitOutput(T& buffer, std::size_t size) {
        if (!buffer.empty()) {
            JST_FATAL("The output buffer should be empty on initialization.");
            return Result::ERROR;
        }

        return buffer.resize(size);
    }

    friend class Instance;
};

}  // namespace Jetstream

#endif
