#include <catch2/catch_test_macros.hpp>

#include <limits>

#include "jetstream/memory/axis.hh"
#include "jetstream/memory/tensor.hh"

using namespace Jetstream;

TEST_CASE("ResolveAxis normalizes regular tensor axes", "[core][memory][axis]") {
    REQUIRE(ResolveAxis(0, 3) == Index{0});
    REQUIRE(ResolveAxis(2, 3) == Index{2});
    REQUIRE(ResolveAxis(-1, 3) == Index{2});
    REQUIRE(ResolveAxis(-3, 3) == Index{0});

    REQUIRE_FALSE(ResolveAxis(0, 0));
    REQUIRE_FALSE(ResolveAxis(3, 3));
    REQUIRE_FALSE(ResolveAxis(-4, 3));
    REQUIRE_FALSE(ResolveAxis(std::numeric_limits<I64>::min(), 3));
    REQUIRE_FALSE(ResolveAxis(std::numeric_limits<I64>::max(), 3));
    REQUIRE_FALSE(ResolveAxis(0, std::numeric_limits<Index>::max()));
}

TEST_CASE("ResolveInsertionAxis normalizes dimension insertion axes",
          "[core][memory][axis]") {
    REQUIRE(ResolveInsertionAxis(0, 2) == Index{0});
    REQUIRE(ResolveInsertionAxis(2, 2) == Index{2});
    REQUIRE(ResolveInsertionAxis(-1, 2) == Index{2});
    REQUIRE(ResolveInsertionAxis(-2, 2) == Index{1});
    REQUIRE(ResolveInsertionAxis(-3, 2) == Index{0});

    REQUIRE(ResolveInsertionAxis(0, 0) == Index{0});
    REQUIRE(ResolveInsertionAxis(-1, 0) == Index{0});
    REQUIRE_FALSE(ResolveInsertionAxis(3, 2));
    REQUIRE_FALSE(ResolveInsertionAxis(-4, 2));
    REQUIRE_FALSE(ResolveInsertionAxis(std::numeric_limits<I64>::min(), 2));
    REQUIRE_FALSE(ResolveInsertionAxis(std::numeric_limits<I64>::max(), 2));
    REQUIRE_FALSE(ResolveInsertionAxis(0, std::numeric_limits<Index>::max()));
}

TEST_CASE("ResolveSignalAxes validates standard signal axes",
           "[core][memory][axis]") {
    Tensor samples(DeviceType::CPU, DataType::F32, {8});

    SignalAxes axes;
    REQUIRE(ResolveSignalAxes(samples, axes) == Result::SUCCESS);
    REQUIRE(axes.sample == Index{0});
    REQUIRE_FALSE(axes.batch);
    REQUIRE_FALSE(axes.channel);

    Tensor multidimensional(DeviceType::CPU, DataType::F32, {2, 3, 8, 4});
    const SignalAxes expected{
        .sample = Index{2},
        .channel = Index{1},
    };
    REQUIRE(SetSignalAxes(multidimensional, expected) == Result::SUCCESS);
    REQUIRE(ResolveSignalAxes(multidimensional, axes) == Result::SUCCESS);
    REQUIRE(axes.sample == expected.sample);
    REQUIRE(axes.batch == expected.batch);
    REQUIRE(axes.channel == expected.channel);

    Tensor batched(DeviceType::CPU, DataType::F32, {5, 2, 3, 8, 4});
    const SignalAxes batchedExpected{
        .sample = Index{3},
        .batch = Index{0},
        .channel = Index{2},
    };
    REQUIRE(SetSignalAxes(batched, batchedExpected) == Result::SUCCESS);
    REQUIRE(ResolveSignalAxes(batched, axes) == Result::SUCCESS);
    REQUIRE(axes.sample == batchedExpected.sample);
    REQUIRE(axes.batch == batchedExpected.batch);
    REQUIRE(axes.channel == batchedExpected.channel);
}

TEST_CASE("ResolveSignalAxes rejects ambiguous or malformed layouts",
           "[core][memory][axis]") {
    SignalAxes axes;

    Tensor missingSample(DeviceType::CPU, DataType::F32, {2, 8});
    REQUIRE(missingSample.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
    REQUIRE(ResolveSignalAxes(missingSample, axes) == Result::ERROR);

    Tensor wrongType(DeviceType::CPU, DataType::F32, {2, 8});
    REQUIRE(wrongType.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(wrongType.setAttribute("batchAxis", I64{0}) == Result::SUCCESS);
    REQUIRE(ResolveSignalAxes(wrongType, axes) == Result::ERROR);

    Tensor outOfRange(DeviceType::CPU, DataType::F32, {2, 8});
    REQUIRE(outOfRange.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
    REQUIRE(ResolveSignalAxes(outOfRange, axes) == Result::ERROR);

    Tensor duplicate(DeviceType::CPU, DataType::F32, {2, 8});
    REQUIRE(duplicate.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(duplicate.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(ResolveSignalAxes(duplicate, axes) == Result::ERROR);

    Tensor valid(DeviceType::CPU, DataType::F32, {2, 8});
    REQUIRE(SetSignalAxes(valid, {.sample = Index{1}, .batch = Index{1}}) ==
            Result::ERROR);
}

TEST_CASE("SetSignalAxes clears standard attributes", "[core][memory][axis]") {
    Tensor tensor(DeviceType::CPU, DataType::F32, {2, 8});
    REQUIRE(SetSignalAxes(tensor, {
        .sample = Index{1},
        .batch = Index{0},
    }) == Result::SUCCESS);
    REQUIRE(tensor.hasAttribute("sampleAxis"));
    REQUIRE(tensor.hasAttribute("batchAxis"));

    REQUIRE(SetSignalAxes(tensor, {}) == Result::SUCCESS);
    REQUIRE_FALSE(tensor.hasAttribute("sampleAxis"));
    REQUIRE_FALSE(tensor.hasAttribute("batchAxis"));
    REQUIRE_FALSE(tensor.hasAttribute("channelAxis"));
}

TEST_CASE("MapSignalAxes remaps and removes roles", "[core][memory][axis]") {
    Tensor implicitSamples(DeviceType::CPU, DataType::F32, {8});
    SignalAxes axes;
    REQUIRE(MapSignalAxes(implicitSamples, {Index{1}}, axes) == Result::SUCCESS);
    REQUIRE(axes.sample == Index{1});

    Tensor input(DeviceType::CPU, DataType::F32, {2, 3, 8});
    REQUIRE(SetSignalAxes(input, {
        .sample = Index{2},
        .batch = Index{0},
        .channel = Index{1},
    }) == Result::SUCCESS);

    REQUIRE(MapSignalAxes(input, {Index{1}, Index{2}, Index{0}}, axes) ==
            Result::SUCCESS);
    REQUIRE(axes.sample == Index{0});
    REQUIRE(axes.batch == Index{1});
    REQUIRE(axes.channel == Index{2});

    REQUIRE(MapSignalAxes(input, {Index{0}, std::nullopt, Index{1}}, axes) ==
            Result::SUCCESS);
    REQUIRE(axes.sample == Index{1});
    REQUIRE(axes.batch == Index{0});
    REQUIRE_FALSE(axes.channel);

    REQUIRE(MapSignalAxes(input, {Index{0}, Index{1}, std::nullopt}, axes) ==
            Result::SUCCESS);
    REQUIRE_FALSE(axes.sample);
    REQUIRE(axes.batch == Index{0});
    REQUIRE(axes.channel == Index{1});

    Tensor withoutSamples(DeviceType::CPU, DataType::F32, {2, 3});
    REQUIRE(SetSignalAxes(withoutSamples, axes) == Result::SUCCESS);
    REQUIRE_FALSE(withoutSamples.hasAttribute("sampleAxis"));
    REQUIRE(withoutSamples.hasAttribute("batchAxis"));
    REQUIRE(withoutSamples.hasAttribute("channelAxis"));

    REQUIRE(MapSignalAxes(withoutSamples, {Index{1}, Index{0}}, axes) ==
            Result::SUCCESS);
    REQUIRE_FALSE(axes.sample);
    REQUIRE(axes.batch == Index{1});
    REQUIRE(axes.channel == Index{0});

    REQUIRE(MapSignalAxes(input, {Index{0}}, axes) == Result::ERROR);

}
