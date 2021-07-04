#ifndef JETSTREAM_ENGINE_H
#define JETSTREAM_ENGINE_H

#include "jetstream/type.hpp"
#include "jetstream/modules/base.hpp"
#include "jetstream/scheduler/base.hpp"

namespace Jetstream {

class Engine {
public:
    explicit Engine(const Policy & defaultPolicy = {Locale::CPU, Launch::SYNC}) :
        defaultPolicy(defaultPolicy) {};

    template<typename T>
    std::shared_ptr<T> add(const std::string & name, const typename T::Config & cfg,
            const Draft & draft) {
        return this->add<T>(name, cfg, draft, defaultPolicy);
    }

    template<typename T>
    std::shared_ptr<T> add(const std::string & name, const typename T::Config & cfg,
            const Draft & draft, const Policy & policy) {
        auto& input = instances[name];

        for (const auto& [key, var] : draft) {
            switch (var.index()) {
                case 0: break;
                case 1: { // DataContainer
                    auto& container = std::get<DataContainer>(var);

                    if (T::inputBlueprint(policy.device)[key] != container) {
                        std::cerr << "[JETSTREAM] Mismatched input and output!" << std::endl;
                        std::cerr << "            "
                                  << "Input{" << name << ", " << key << "} << "
                                  << "Output{Inline}"
                                  << std::endl;
                        JETSTREAM_CHECK_THROW(Result::ERROR);
                    }

                    input.inputs[key] = container;
                    break;
                }
                case 2: { // Tap
                    auto& [out_name, out_port] = std::get<Tap>(var);
                    auto& output = instances[out_name];
                    const auto& container = output.mod->output(out_port);

                    if (T::inputBlueprint(policy.device)[key] != container) {
                        std::cerr << "[JETSTREAM] Mismatched input and output!" << std::endl;
                        std::cerr << "            "
                                  << "Input{" << name << ", " << key << "} << "
                                  << "Output{" << out_name << ", " << out_port << "}"
                                  << std::endl;
                        JETSTREAM_CHECK_THROW(Result::ERROR);
                    }

                    input.inputs[key] = container;
                    input.deps.push_back(output.run);
                    break;
                }
            }
        }

        input.pol = policy;
        input.mod = ModuleFactory<T>(input.pol.device, cfg, input.inputs);
        input.run = SchedulerFactory(input.pol.mode, input.mod, input.deps);

        return this->get<T>(name);
    }

    template<typename T>
    std::shared_ptr<T> get(const std::string & name) {
        return std::static_pointer_cast<T>(instances[name].mod);
    }

    Result compute();
    Result present();

private:
    typedef struct {
        Policy pol;
        Dependencies deps;
        Connections inputs;
        std::shared_ptr<Module> mod;
        std::shared_ptr<Scheduler> run;
    } Instance;

    const Policy defaultPolicy;
    std::map<std::string, Instance> instances;

    std::mutex m;
    std::condition_variable access;
    std::atomic<bool> waiting{false};
    std::atomic<bool> lock{false};
};

} // namespace Jetstream

#endif
