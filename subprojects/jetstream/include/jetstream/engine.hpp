#ifndef JETSTREAM_ENGINE_H
#define JETSTREAM_ENGINE_H

#include "jetstream/type.hpp"
#include "jetstream/factory.hpp"
#include "jetstream/scheduler/base.hpp"

namespace Jetstream {

class Engine {
public:
    explicit Engine(const Policy & defaultPolicy = {Locale::CUDA, Launch::ASYNC}) :
        defaultPolicy(defaultPolicy) {};

    template<typename T>
    Result add(const std::string & name, const typename T::Config & cfg,
            const VirtualManifest & manifest) {
        return this->add<T>(name, cfg, manifest, defaultPolicy);
    }

    template<typename T>
    Result add(const std::string & name, const typename T::Config & cfg,
            const VirtualManifest & manifest, const Policy & policy) {
        auto& worker = stream[name];

        for (const auto& [key, var] : manifest) {
            switch (var.index()) {
                case 0: break;
                case 1: { // Variant
                    worker.inputs[key] = std::get<Variant>(var);
                    break;
                }
                case 2: { // Tap
                    auto tap = std::get<Tap>(var);
                    auto& wrk = stream[tap.module];
                    worker.inputs[key] = wrk.mod->output(tap.port);
                    worker.deps.push_back(wrk.run);
                    break;
                }
            }
        }

        worker.pol = policy;
        worker.mod = Factory<T>(worker.pol.device, cfg, worker.inputs);
        worker.run = Factory(worker.pol.mode, worker.mod, worker.deps);

        return Result::SUCCESS;
    }

    template<typename T>
    std::shared_ptr<T> get(const std::string & name) {
        return std::static_pointer_cast<T>(stream[name].mod);
    }

    Result compute();
    Result present();

private:
    typedef struct {
        Policy pol;
        Manifest inputs;
        Dependencies deps;
        std::shared_ptr<Module> mod;
        std::shared_ptr<Scheduler> run;
    } Worker;

    const Policy defaultPolicy;
    std::map<std::string, Worker> stream;

    std::mutex m;
    std::condition_variable access;
    std::atomic<bool> waiting{false};
    std::atomic<bool> lock{false};
};

} // namespace Jetstream

#endif
