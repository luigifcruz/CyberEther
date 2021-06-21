#ifndef JETSTREAM_MODULE_H
#define JETSTREAM_MODULE_H

#include "scheduler/sync.hpp"
#include "scheduler/async.hpp"

namespace Jetstream {

class Module : public Async, public Sync {
public:
    explicit Module(const Policy&);
    virtual ~Module();

    Result compute();
    Result barrier();
    Result present();

protected:
    const Launch launch;

    virtual Result underlyingPresent() = 0;
};

} // namespace Jetstream

#endif
