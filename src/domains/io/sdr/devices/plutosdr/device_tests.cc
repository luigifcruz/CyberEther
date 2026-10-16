#include <catch2/catch_test_macros.hpp>

#include <limits>

#include "domains/io/sdr/devices/plutosdr/sweep_buffer.hh"

using namespace Jetstream;

TEST_CASE("Pluto sweep transfers hold complete FFT captures", "[devices][sdr][plutosdr]") {
    const auto small = Radio::PlutoSweepBufferLayout::Create(800, 1);
    REQUIRE(small);
    CHECK(small->captureSamples == 800);
    CHECK(small->batchSamples == 800);
    CHECK(small->transferSamples == 1024);

    const auto averaged = Radio::PlutoSweepBufferLayout::Create(2000, 4);
    REQUIRE(averaged);
    CHECK(averaged->captureSamples == 8000);
    CHECK(averaged->batchSamples == 8000);
    CHECK(averaged->transferSamples == 8000);

    const auto fine = Radio::PlutoSweepBufferLayout::Create(20000, 4);
    REQUIRE(fine);
    CHECK(fine->captureSamples == 80000);
    CHECK(fine->batchSamples == 20000);
    CHECK(fine->transferSamples == 20000);
}

TEST_CASE("Pluto splits long dwells only between complete FFTs", "[devices][sdr][plutosdr]") {
    const auto large = Radio::PlutoSweepBufferLayout::Create(65536, 64);
    REQUIRE(large);
    CHECK(large->captureSamples == 4194304);
    CHECK(large->batchSamples == 65536);
    CHECK(large->transferSamples == 65536);

    const auto unaligned = Radio::PlutoSweepBufferLayout::Create(65537, 17);
    REQUIRE(unaligned);
    CHECK(unaligned->captureSamples == 1114129);
    CHECK(unaligned->batchSamples == 65537);
    CHECK(unaligned->transferSamples == 65552);

    const auto grouped = Radio::PlutoSweepBufferLayout::Create(800, 64);
    REQUIRE(grouped);
    CHECK(grouped->captureSamples == 51200);
    CHECK(grouped->batchSamples == 8000);
    CHECK(grouped->transferSamples == 8000);
    CHECK(grouped->captureSamples % grouped->batchSamples == 3200);
}

TEST_CASE("Pluto rejects invalid sweep allocations", "[devices][sdr][plutosdr]") {
    CHECK_FALSE(Radio::PlutoSweepBufferLayout::Create(0, 1));
    CHECK_FALSE(Radio::PlutoSweepBufferLayout::Create(800, 0));
    CHECK_FALSE(Radio::PlutoSweepBufferLayout::Create(1048577, 1));
    CHECK_FALSE(Radio::PlutoSweepBufferLayout::Create(65536, std::numeric_limits<U64>::max()));
    CHECK_FALSE(Radio::PlutoSweepBufferLayout::Create(1, std::numeric_limits<U64>::max()));
}
