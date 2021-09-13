#ifndef JETSTREAM_BASE_H
#define JETSTREAM_BASE_H

#include "jetstream/module.hpp"
#include "jetstream/modifiers/sync.hpp"
#include "jetstream/modifiers/async.hpp"

namespace Jetstream {

template<template<class> class X>
class Loop : public Module {
public:
    struct Config {};
    struct Input {};

    Loop() {};
    Loop(const Config &, const Input &) {};

    static std::shared_ptr<Loop<X>> New() {
        return std::make_shared<Loop<X>>();
    }

    template<template<class> class Y>
    static std::shared_ptr<Loop<X>> New(const std::shared_ptr<Loop<Y>> & node) {
        return node->template add<Loop<X>>("modifier", {}, {});
    }

    template<typename T>
    std::shared_ptr<T> add(const std::string & name, const typename T::Config & config,
            const typename T::Input & input) {
        auto mod = std::make_shared<X<T>>(config, input);
        blocks.push_back({ name, mod });
        return mod;
    }

    Result compute() {
        DEBUG_PUSH("compute_wait");

        std::unique_lock<std::mutex> sync(m);
        access.wait(sync, [=]{ return !waiting; });

        DEBUG_POP();
        DEBUG_PUSH("compute");

        Result err = Result::SUCCESS;
        for (const auto & [name, mod] : blocks) {
            DEBUG_PUSH(name + "_compute");
            if ((err = mod->compute()) != Result::SUCCESS) {
                return err;
            }
            DEBUG_POP();
        }

        for (const auto & [name, mod] : blocks) {
            DEBUG_PUSH(name + "_barrier");
            if ((err = mod->barrier()) != Result::SUCCESS) {
                return err;
            }
            DEBUG_POP();
        }

        DEBUG_POP();

        return err;
    }

    Result present() {
        DEBUG_PUSH("present_wait");

        waiting = true;
        Result err = Result::SUCCESS;
        {
            const std::unique_lock<std::mutex> lock(m);

            DEBUG_POP();
            DEBUG_PUSH("present");

            for (const auto & [name, mod] : blocks) {
                DEBUG_PUSH(name + "_present");
                if ((err = mod->present()) != Result::SUCCESS) {
                    return err;
                }
                DEBUG_POP();
            }
        }
        waiting = false;
        access.notify_one();

        DEBUG_POP();

        return err;
    }

protected:
    struct Block {
        std::string name;
        std::shared_ptr<Module> mod;
    };
    std::vector<Block> blocks;
    std::condition_variable access;
    bool waiting = false;
    std::mutex m;
};

} // namespace Jetstream

#endif
