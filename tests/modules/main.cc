#include "test_main.hh"

#include "jetstream/backend/base.hh"
#include "jetstream/logger.hh"

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(4);

    return TestSupport::Run(argc, argv);
}
