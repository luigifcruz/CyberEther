#include "jetstream/base.hpp"

namespace Jetstream {

Engine::Engine(const Policy & policy) : defaultPolicy(policy) {
}

Engine::~Engine() {
}

Result Engine::begin() {
    lock.store(true);

    // code

    return Result::SUCCESS;
}

Result Engine::end() {
    // code

    lock.store(false);
    return Result::SUCCESS;
}

Result Engine::add(const std::string name, const std::unique_ptr<Module> & mod) {
    return this->add(name, mod, defaultPolicy);
}

Result Engine::add(const std::string name, const std::unique_ptr<Module> & mod, const Policy pol) {
    // code

    return Result::SUCCESS;
}

Result Engine::remove(const std::string name) {
    // code

    return Result::SUCCESS;
}

template<typename T>
std::weak_ptr<T> Engine::get(const std::string name) {
    return static_cast<T>(stream[name].mod);
}

Result Engine::compute() {
    DEBUG_PUSH("compute_wait");

    std::unique_lock<std::mutex> sync(m);
    access.wait(sync, [=]{ return !waiting; });

    DEBUG_POP();
    DEBUG_PUSH("compute");

    for (const auto& transform : stream) {
        auto result = transform->compute();
        if (result != Result::SUCCESS) {
            return result;
        }
    }

    for (const auto& transform : *this) {
        auto result = transform->barrier();
        if (result != Result::SUCCESS) {
            return result;
        }
    }

    DEBUG_POP();

    return Result::SUCCESS;
}

Result Engine::present() {
    DEBUG_PUSH("present_wait");

    waiting = true;
    {
        const std::scoped_lock<std::mutex> lock(m);

        DEBUG_POP();
        DEBUG_PUSH("present");

        for (const auto& transform : *this) {
            auto result = transform->present();
            if (result != Result::SUCCESS) {
                return result;
            }
        }
    }
    waiting = false;
    access.notify_one();

    DEBUG_POP();

    return Result::SUCCESS;
}

} // namespace Jetstream
