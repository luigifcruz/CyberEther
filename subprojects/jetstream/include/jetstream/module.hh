#ifndef JETSTREAM_MODULE_HH
#define JETSTREAM_MODULE_HH

#include "jetstream/base.hh"

namespace Jetstream {

class JETSTREAM_API Module {
public:
    virtual ~Module() = default;

protected:
    virtual constexpr const Result compute() {
        return Result::SUCCESS;
    }

    virtual constexpr const Result present() {
        return Result::SUCCESS;
    }

    friend class Instance;
};

}  // namespace Jetstream

#endif
