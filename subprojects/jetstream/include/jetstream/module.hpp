#ifndef JETSTREAM_MODULE_H
#define JETSTREAM_MODULE_H

#include "jetstream/type.hpp"

namespace Jetstream {

class Module {
public:
    virtual ~Module() = default;
    virtual Result compute() = 0;
    virtual Result present() = 0;

    Variant output(const std::string & name) {
        return out_manifest[name];
    }

protected:
    Manifest out_manifest;
};

} // namespace Jetstream

#endif
