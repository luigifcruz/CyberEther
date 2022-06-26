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
    return Get()->pollEvents();
}

const Result Synchronize() {
    return Get()->synchronize();
}

const bool KeepRunning() {
    return Get()->keepRunning();
}

}  // namespace Jetstream::Render 
