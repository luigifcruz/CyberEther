#include "jetstream/render/base.hh"

namespace Jetstream::Render {

std::shared_ptr<Window>& Get(const bool& safe) {
    static std::shared_ptr<Window> instance;

    if (!instance && safe) {
        JST_FATAL("Render not initialized yet.");
        throw Result::ERROR;
    }

    return instance;
}

const Result Create() {
    return Get()->create();
}

const Result Destroy() {
    return Get()->destroy();
}

const Result Begin() {
    return Get()->begin();
}

const Result End() {
    return Get()->end();
}

const Result PollEvents() {
    if (!Get(false)) {
        return Result::SUCCESS;
    }

    return Get()->pollEvents();
}

const Result Synchronize() {
    return Get()->synchronize();
}

const bool KeepRunning() {
    if (!Get(false)) {
        return true;
    }

    return Get()->keepRunning();
}

}  // namespace Jetstream::Render 