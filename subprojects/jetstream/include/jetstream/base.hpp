#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

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

class Engine : public Graph {
public:
    Result compute();
    Result present();

private:
    std::mutex m;
    std::condition_variable access;
    std::atomic<bool> waiting{false};
};

} // namespace Jetstream

#endif
