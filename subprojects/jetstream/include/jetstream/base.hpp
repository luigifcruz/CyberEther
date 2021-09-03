#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include "jetstream/module.hpp"
#include "jetstream/asyncify.hpp"

namespace Jetstream {

class Loop : public Module {
public:
    struct Config {
        bool async = false;
    };
    struct Input {};

    Loop(const Config & config, const Input &) : async(config.async) {};

    static std::shared_ptr<Loop> Sync(std::shared_ptr<Loop> node = nullptr) {
        if (node) {
            return node->add<Loop>("sync", {}, {});
        }
        return std::make_shared<Loop>(Config{}, Input{});
    }

    static std::shared_ptr<Loop> Async(std::shared_ptr<Loop> node = nullptr) {
        if (node) {
            return node->add<Loop>("async", {true}, {});
        }
        return std::make_shared<Loop>(Config{true}, Input{});
    }

    template<typename T>
    std::shared_ptr<T> add(std::string name, typename T::Config config, typename T::Input input) {
        if (async) {
            auto mod = std::make_shared<Asyncify<T>>(config, input);
            blocks.push_back({ name, mod });
            return mod->sync();
        }
        auto mod = std::make_shared<T>(config, input);
        blocks.push_back({ name, mod });
        return mod;
    }

    Result compute() final;
    Result present() final;

protected:
    struct Block {
        std::string name;
        std::shared_ptr<Module> mod;
    };
    bool async;
    std::mutex m;
    std::vector<Block> blocks;
    std::condition_variable access;
    std::atomic<bool> waiting{false};
};

} // namespace Jetstream

#endif
