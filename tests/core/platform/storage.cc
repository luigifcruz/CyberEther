#include <catch2/catch_test_macros.hpp>

#include "jetstream/platform.hh"

using namespace Jetstream;

TEST_CASE("Platform persistent storage initializes", "[core][platform][storage]") {
    REQUIRE(Platform::InitializePersistentStorage() == Result::SUCCESS);
}
