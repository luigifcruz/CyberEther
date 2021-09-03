#ifndef JETSTREAM_MODULE_H
#define JETSTREAM_MODULE_H

#include "jetstream/type.hpp"

namespace Jetstream {

class Module {
public:
    virtual ~Module() = default;

    virtual Result compute() {
        return Result::SUCCESS;
    }

    virtual Result present() {
        return Result::SUCCESS;
    }

    virtual Result barrier() {
        return Result::SUCCESS;
    }

protected:
    virtual Result underlyingCompute() {
        return Result::SUCCESS;
    }

    virtual Result underlyingPresent() {
        return Result::SUCCESS;
    }
};

} // namespace Jetstream

#endif
