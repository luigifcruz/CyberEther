#include "test_main.hh"

#include "jetstream/logger.hh"

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(0);
    return TestSupport::Run(argc, argv);
}
