#include "jetstream/memory/axis.hh"

#include <array>
#include <limits>
#include <string>

#include "jetstream/memory/tensor.hh"

namespace Jetstream {

namespace {

Result ReadAxis(const Tensor& tensor,
                const std::string_view name,
                std::optional<Index>& axis) {
    axis.reset();

    const std::string key(name);
    if (!tensor.hasAttribute(key)) {
        return Result::SUCCESS;
    }

    const std::any value = tensor.attribute(key);
    const auto* resolvedAxis = std::any_cast<Index>(&value);
    if (resolvedAxis == nullptr) {
        JST_ERROR("[MEMORY:AXIS] Attribute '{}' must have type Index.", name);
        return Result::ERROR;
    }
    if (*resolvedAxis >= tensor.rank()) {
        JST_ERROR("[MEMORY:AXIS] Attribute '{}' axis {} is out of range for rank {}.",
                  name, *resolvedAxis, tensor.rank());
        return Result::ERROR;
    }

    axis = *resolvedAxis;
    return Result::SUCCESS;
}

Result ValidateAxes(const Tensor& tensor,
                    const SignalAxes& axes,
                    const bool requireSample) {
    if (requireSample && !axes.sample) {
        JST_ERROR("[MEMORY:AXIS] Signal tensor is missing sampleAxis metadata.");
        return Result::ERROR;
    }

    const std::array roles{
        std::pair{SampleAxisAttribute, axes.sample},
        std::pair{BatchAxisAttribute, axes.batch},
        std::pair{ChannelAxisAttribute, axes.channel},
    };

    for (std::size_t i = 0; i < roles.size(); ++i) {
        const auto& [name, axis] = roles[i];
        if (!axis) {
            continue;
        }
        if (*axis >= tensor.rank()) {
            JST_ERROR("[MEMORY:AXIS] Attribute '{}' axis {} is out of range for rank {}.",
                      name, *axis, tensor.rank());
            return Result::ERROR;
        }
        for (std::size_t j = i + 1; j < roles.size(); ++j) {
            if (roles[j].second && *axis == *roles[j].second) {
                JST_ERROR("[MEMORY:AXIS] Attributes '{}' and '{}' cannot use axis {}.",
                          name, roles[j].first, *axis);
                return Result::ERROR;
            }
        }
    }

    return Result::SUCCESS;
}

bool HasSignalAxes(const Tensor& tensor) {
    return tensor.hasAttribute(std::string(SampleAxisAttribute)) ||
           tensor.hasAttribute(std::string(BatchAxisAttribute)) ||
           tensor.hasAttribute(std::string(ChannelAxisAttribute));
}

}  // namespace

std::optional<Index> ResolveAxis(const I64 axis, const Index rank) {
    if (rank == 0 || rank > static_cast<Index>(std::numeric_limits<I64>::max())) {
        return std::nullopt;
    }

    const I64 signedRank = static_cast<I64>(rank);
    const I64 resolvedAxis = axis < 0 ? signedRank + axis : axis;
    if (resolvedAxis < 0 || resolvedAxis >= signedRank) {
        return std::nullopt;
    }

    return static_cast<Index>(resolvedAxis);
}

std::optional<Index> ResolveInsertionAxis(const I64 axis, const Index rank) {
    if (rank > static_cast<Index>(std::numeric_limits<I64>::max())) {
        return std::nullopt;
    }

    const I64 signedRank = static_cast<I64>(rank);
    if (axis >= 0) {
        if (axis > signedRank) {
            return std::nullopt;
        }
        return static_cast<Index>(axis);
    }

    const I64 resolvedAxisMinusOne = signedRank + axis;
    if (resolvedAxisMinusOne < -1) {
        return std::nullopt;
    }

    return static_cast<Index>(resolvedAxisMinusOne + 1);
}

Result ResolveSignalAxes(const Tensor& tensor, SignalAxes& axes) {
    axes = {};

    JST_CHECK(ReadAxis(tensor, SampleAxisAttribute, axes.sample));
    JST_CHECK(ReadAxis(tensor, BatchAxisAttribute, axes.batch));
    JST_CHECK(ReadAxis(tensor, ChannelAxisAttribute, axes.channel));
    if (!axes.sample && tensor.rank() == 1) {
        axes.sample = Index{0};
    }
    JST_CHECK(ValidateAxes(tensor, axes, true));

    return Result::SUCCESS;
}

Result SetSignalAxes(Tensor& tensor, const SignalAxes& axes) {
    JST_CHECK(ValidateAxes(tensor, axes, false));

    const auto setOrRemove = [&](const std::string_view name,
                                 const std::optional<Index>& axis) -> Result {
        const std::string key(name);
        if (axis) {
            return tensor.setAttribute(key, *axis);
        }
        return tensor.removeAttribute(key);
    };

    JST_CHECK(setOrRemove(SampleAxisAttribute, axes.sample));
    JST_CHECK(setOrRemove(BatchAxisAttribute, axes.batch));
    JST_CHECK(setOrRemove(ChannelAxisAttribute, axes.channel));

    return Result::SUCCESS;
}

Result MapSignalAxes(const Tensor& tensor,
                     const AxisMap& axisMap,
                     SignalAxes& axes) {
    axes = {};

    if (axisMap.size() != tensor.rank()) {
        JST_ERROR("[MEMORY:AXIS] Axis map size {} does not match tensor rank {}.",
                  axisMap.size(), tensor.rank());
        return Result::ERROR;
    }
    const bool hasSignalAxes = HasSignalAxes(tensor);
    if (!hasSignalAxes && tensor.rank() != 1) {
        return Result::SUCCESS;
    }

    SignalAxes inputAxes;
    JST_CHECK(ReadAxis(tensor, SampleAxisAttribute, inputAxes.sample));
    JST_CHECK(ReadAxis(tensor, BatchAxisAttribute, inputAxes.batch));
    JST_CHECK(ReadAxis(tensor, ChannelAxisAttribute, inputAxes.channel));
    if (!hasSignalAxes) {
        inputAxes.sample = Index{0};
    }
    JST_CHECK(ValidateAxes(tensor, inputAxes, false));

    const auto mapAxis = [&](const std::optional<Index>& inputAxis,
                             std::optional<Index>& outputAxis) {
        if (inputAxis) {
            outputAxis = axisMap[*inputAxis];
        }
    };

    mapAxis(inputAxes.sample, axes.sample);
    mapAxis(inputAxes.batch, axes.batch);
    mapAxis(inputAxes.channel, axes.channel);

    const std::array mappedAxes{axes.sample, axes.batch, axes.channel};
    for (std::size_t i = 0; i < mappedAxes.size(); ++i) {
        if (!mappedAxes[i]) {
            continue;
        }
        for (std::size_t j = i + 1; j < mappedAxes.size(); ++j) {
            if (mappedAxes[j] && mappedAxes[i] == mappedAxes[j]) {
                JST_ERROR("[MEMORY:AXIS] Axis map merges distinct signal roles.");
                return Result::ERROR;
            }
        }
    }

    return Result::SUCCESS;
}

AxisMap RightAlignedAxisMap(const Index inputRank, const Index outputRank) {
    AxisMap axisMap(inputRank);
    const Index offset = outputRank - inputRank;
    for (Index axis = 0; axis < inputRank; ++axis) {
        axisMap[axis] = offset + axis;
    }
    return axisMap;
}

AxisMap IdentityAxisMap(const Index rank) {
    AxisMap axisMap(rank);
    for (Index axis = 0; axis < rank; ++axis) {
        axisMap[axis] = axis;
    }
    return axisMap;
}

Result MergeBroadcastSignalAxes(const Tensor& tensorA,
                                const Tensor& tensorB,
                                Tensor& output) {
    SignalAxes axesA;
    SignalAxes axesB;
    JST_CHECK(MapSignalAxes(
        tensorA, RightAlignedAxisMap(tensorA.rank(), output.rank()), axesA));
    JST_CHECK(MapSignalAxes(
        tensorB, RightAlignedAxisMap(tensorB.rank(), output.rank()), axesB));

    SignalAxes outputAxes;
    const auto mergeRole = [](const std::optional<Index>& axisA,
                              const std::optional<Index>& axisB,
                              std::optional<Index>& outputAxis) -> Result {
        if (axisA && axisB && axisA != axisB) {
            JST_ERROR("[MEMORY:AXIS] Signal roles map to conflicting output axes.");
            return Result::ERROR;
        }
        outputAxis = axisA ? axisA : axisB;
        return Result::SUCCESS;
    };

    JST_CHECK(mergeRole(axesA.sample, axesB.sample, outputAxes.sample));
    JST_CHECK(mergeRole(axesA.batch, axesB.batch, outputAxes.batch));
    JST_CHECK(mergeRole(axesA.channel, axesB.channel, outputAxes.channel));

    return SetSignalAxes(output, outputAxes);
}

}  // namespace Jetstream
