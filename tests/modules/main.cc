#include <catch2/catch_session.hpp>

#include "jetstream/backend/base.hh"
#include "jetstream/logger.hh"

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(4);

    const int result = Catch::Session().run(argc, argv);
    (void)Jetstream::Backend::DestroyAll();
    return result;
}
