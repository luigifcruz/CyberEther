#include "jetstream/base.hpp"

namespace Jetstream {

Module::Module(Policy& p) : Async(p.deps), Sync(p.deps), launch(p.launch) {
    switch (launch) {
        case Launch::ASYNC:
            JETSTREAM_CHECK_THROW(Async::start());
            break;
        case Launch::SYNC:
            JETSTREAM_CHECK_THROW(Sync::start());
            break;
    }
}

Module::~Module() {
    switch (launch) {
        case Launch::ASYNC:
            Async::end();
            break;
        case Launch::SYNC:
            Sync::end();
            break;
    }
}

Result Module::compute() {
    switch (launch) {
        case Launch::ASYNC:
            return Async::compute();
        case Launch::SYNC:
            return Sync::compute();
    }
}

Result Module::barrier() {
    switch (launch) {
        case Launch::ASYNC:
            return Async::barrier();
        case Launch::SYNC:
            return Sync::barrier();
    }
}

Result Module::present() {
    return this->underlyingPresent();
}

} // namespace Jetstream
