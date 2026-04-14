#define CATCH_CONFIG_RUNNER

#include <string>
#include <thread>

#include <catch2/catch_all.hpp>

#include "jetstream/types.hh"
#include "jetstream/tools/snapshot.hh"

using namespace Jetstream;

int main(int argc, char* argv[]) {
    return Catch::Session().run(argc, argv);
}

TEST_CASE("Snapshot stores and reads trivial values", "[snapshot][trivial]") {
    Tools::Snapshot<U64> value(42);

    REQUIRE(value.get() == 42);

    value.publish(99);

    REQUIRE(value.get() == 99);
}

TEST_CASE("Snapshot stores and reads snapshot values", "[snapshot][snapshot]") {
    Tools::Snapshot<std::string> value(std::string("alpha"));

    REQUIRE(value.get() == "alpha");

    value.publish(std::string("beta"));

    REQUIRE(value.get() == "beta");
}

TEST_CASE("Snapshot supports cross-thread publication", "[snapshot][threading]") {
    Tools::Snapshot<U64> value(0);

    std::thread producer([&value]() {
        for (U64 i = 1; i <= 1024; ++i) {
            value.publish(i);
        }
    });

    producer.join();

    REQUIRE(value.get() == 1024);
}
