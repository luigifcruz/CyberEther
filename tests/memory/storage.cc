#include <sstream>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "jetstream/memory/base.hh"

using namespace Jetstream;

TEST_CASE("Storage Class Tests", "[Storage]") {
    SECTION("Default Constructor") {
        Tensor<Device::CPU, F32> storage;

        REQUIRE(storage.root_device() == Device::CPU);
        REQUIRE(storage.compatible_devices().size() >= 1);
        REQUIRE(storage.compatible_devices().contains(Device::CPU));
        REQUIRE(storage.references() == 1);
        REQUIRE(storage.attributes().empty());
    }

    // TODO: Add more tests.

#if defined(JETSTREAM_BACKEND_VULKAN_AVAILABLE)
    SECTION("Clone") {
        Tensor<Device::Vulkan, F32> storage({1}, true);

        REQUIRE(storage.device() == Device::Vulkan);
        REQUIRE(storage.root_device() == Device::Vulkan);
        REQUIRE(storage.references() == 1);
        REQUIRE(storage.compatible_devices().contains(Device::Vulkan));

        Tensor<Device::CPU, F32> cloned_storage(storage);

        REQUIRE(cloned_storage.device() == Device::CPU);
        REQUIRE(cloned_storage.root_device() == Device::Vulkan);
        REQUIRE(cloned_storage.references() == 2);
        REQUIRE(cloned_storage.compatible_devices().contains(Device::Vulkan));
        REQUIRE(cloned_storage.compatible_devices().contains(Device::CPU));
    }
#endif
}

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(4);

    return Catch::Session().run(argc, argv);
}