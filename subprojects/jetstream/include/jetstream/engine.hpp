#ifndef JETSTREAM_ENGINE_H
#define JETSTREAM_ENGINE_H

#include "jetstream/type.hpp"
#include "jetstream/scheduler/base.hpp"

namespace Jetstream {

class Engine {
public:
    explicit Engine(const Policy & defaultPolicy = {Locale::CPU, Launch::SYNC});
    ~Engine();

    Result add(const std::string name, const std::unique_ptr<Module> & mod);
    Result add(const std::string name, const std::unique_ptr<Module> & mod, const Policy policy);
    Result remove(const std::string name);

    template<typename T>
    std::weak_ptr<T> get(const std::string name);

    Result begin();
    Result end();

    Result compute();
    Result present();

private:
    typedef struct {
        Policy policy;
        std::unique_ptr<Module> mod;
        std::unique_ptr<Scheduler> scheduler;
    } worker;

    const Policy defaultPolicy;
    std::map<std::string, worker> stream;

    std::mutex m;
    std::condition_variable access;
    std::atomic<bool> waiting{false};
    std::atomic<bool> lock{false};
};

} // namespace Jetstream

#endif
