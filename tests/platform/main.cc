#include <catch2/catch_session.hpp>

#include "jetstream/logger.hh"

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(0);
    return Catch::Session().run(argc, argv);
}
