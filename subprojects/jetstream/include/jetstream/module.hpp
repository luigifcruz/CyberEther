#ifndef JETSTREAM_MODULE_H
#define JETSTREAM_MODULE_H

#include "jetstream/type.hpp"

namespace Jetstream {

class Module {
public:
    virtual ~Module() = default;
    virtual Result compute() = 0;
    virtual Result present() = 0;
};

} // namespace Jetstream

#endif
