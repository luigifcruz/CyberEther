#ifndef JETSTREAM_MODULE_H
#define JETSTREAM_MODULE_H

#include "jetstream/type.hpp"

namespace Jetstream {

class Module {
public:
    virtual ~Module() = default;

    virtual constexpr const Feature feature() const {
        return Feature::NONE;
    }

    virtual constexpr const Capability capability() const {
        return Capability::NONE;
    }

protected:
    virtual constexpr const Result compute() {
        return Result::SUCCESS;
    }

    virtual constexpr const Result present() {
        return Result::SUCCESS;
    }

    friend class Instance;
};

} // namespace Jetstream

#endif
